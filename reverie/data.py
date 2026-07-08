"""Data pipeline for ProsQA-style latent-planning tasks.

The Rust generator (`data-gen/`) emits the *abstract* problem as JSONL:
    {id, entities:[name...], edges:[[a,b]...], source, candidates:[c0,c1],
     answer, gold_path:[...], n_hops, n_distractors}

This module owns the *text* side: a compact word-level vocabulary (concept names
are atomic tokens, so no BPE), and rendering of each instance into token id
sequences for each supervision mode:

  * ``nocot``  — [facts, query, <bot>] -> answer            (answer CE only)
  * ``cot``    — [facts, query, <bot>] -> steps -> answer   (CE on steps+answer)
  * ``latent`` — [facts, query, <bot>]; the model appends K continuous thoughts
                 internally, then reads the single answer token. We also carry
                 the gold path (for trajectory distillation) and ``n_hops`` (the
                 teacher's per-instance reasoning depth, for halt supervision).

The answer is a single concept token, which is what makes the per-depth answer
read-out a single batched matmul (see reverie/latent.py).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

# --- concept-name generator: MUST match data-gen/src/main.rs::concept_name ----
_CONS = "bcdfghjklmnpqrstvwxz"  # 20
_VOWELS = "aeiou"  # 5


def concept_name(idx: int) -> str:
    s = []
    while True:
        s.append(_CONS[idx % len(_CONS)])
        idx //= len(_CONS)
        s.append(_VOWELS[idx % len(_VOWELS)])
        idx //= len(_VOWELS)
        if idx == 0:
            break
    return "".join(s)


# --- vocabulary ---------------------------------------------------------------
SPECIALS = ["<pad>", "<bos>", "<eos>", "<bot>", "<eot>", "<q>", "<ans>"]
FUNC = ["every", "is", "a", "or", "."]
PAD, BOS, EOS, BOT, EOT, QMARK, ANS = range(7)


@dataclasses.dataclass(frozen=True)
class Vocab:
    tok2id: dict[str, int]
    id2tok: list[str]
    max_concepts: int

    @property
    def size(self) -> int:
        return len(self.id2tok)

    def concept_id(self, entity_index: int) -> int:
        return self.tok2id[concept_name(entity_index)]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.tok2id[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2tok[i] for i in ids)


def build_vocab(max_concepts: int) -> Vocab:
    toks = list(SPECIALS) + list(FUNC) + [concept_name(i) for i in range(max_concepts)]
    tok2id = {t: i for i, t in enumerate(toks)}
    assert len(tok2id) == len(toks), "duplicate token in vocab"
    return Vocab(tok2id=tok2id, id2tok=toks, max_concepts=max_concepts)


# --- rendering ----------------------------------------------------------------
@dataclasses.dataclass
class Rendered:
    """One tokenized instance, mode-agnostic fields plus per-mode token lists."""

    prompt: list[int]        # facts + query + <bot>  (everything up to reasoning)
    steps: list[int]         # gold reasoning rendered as tokens (for CoT)
    answer: int              # single concept token id (the correct candidate)
    path_targets: list[int]  # concept token id for each hop node (for distillation)
    n_hops: int
    n_distractors: int


def render(inst: dict, vocab: Vocab) -> Rendered:
    ents = inst["entities"]
    cid = lambda i: vocab.tok2id[ents[i]]  # entity index -> token id

    # facts: "every A is a B ." for each edge, shuffled order is already applied
    # by the generator. We keep the generator's order (deterministic).
    facts: list[int] = []
    for a, b in inst["edges"]:
        facts += [vocab.tok2id["every"], cid(a), vocab.tok2id["is"],
                  vocab.tok2id["a"], cid(b), vocab.tok2id["."]]

    # query: <q> is SOURCE a C0 or C1 ?   (rendered compactly with <q> marker)
    c0, c1 = inst["candidates"]
    query = [QMARK, vocab.tok2id["is"], cid(inst["source"]), vocab.tok2id["a"],
             cid(c0), vocab.tok2id["or"], cid(c1), vocab.tok2id["."]]

    prompt = [BOS] + facts + query + [BOT]

    # gold steps as text (for the CoT baseline): "every v0 is a v1 ." per hop edge
    path = inst["gold_path"]  # [source, v1, ..., answer], length n_hops+1
    steps: list[int] = []
    for u, v in zip(path[:-1], path[1:]):
        steps += [vocab.tok2id["every"], cid(u), vocab.tok2id["is"],
                  vocab.tok2id["a"], cid(v), vocab.tok2id["."]]

    # path targets for trajectory distillation: the concept reached after each hop
    # (path[1..]) — thought t should decode to path[t].
    path_targets = [cid(v) for v in path[1:]]

    return Rendered(
        prompt=prompt,
        steps=steps,
        answer=cid(inst["answer"]),
        path_targets=path_targets,
        n_hops=inst["n_hops"],
        n_distractors=inst["n_distractors"],
    )


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def global_lengths(insts: list[dict], vocab: Vocab) -> tuple[int, int]:
    """Max prompt length and max CoT-sequence length over a dataset, so all
    batches can be padded to a single shape (compile once)."""
    r = [render(i, vocab) for i in insts]
    sp = max(len(x.prompt) for x in r)
    sc = max(len(x.prompt) + len(x.steps) + 3 for x in r)  # +ANS +answer +EOS
    return sp, sc


# --- batching -----------------------------------------------------------------
@dataclasses.dataclass
class Batch:
    """Padded arrays for a batch. Shapes documented per field.

    ``prompt_ids`` [B, Sp] with ``prompt_mask`` [B, Sp] (1 = real token).
    ``answer`` [B] single target token id.
    ``path_targets`` [B, K] gold concept per hop (padded with -1), ``path_len`` [B].
    ``cot_ids``/``cot_mask`` [B, Sc] = prompt+steps+answer for the CoT baseline;
        ``cot_loss_mask`` [B, Sc] marks the positions to score (steps+answer).
    """

    prompt_ids: np.ndarray
    prompt_mask: np.ndarray
    answer: np.ndarray
    path_targets: np.ndarray
    path_len: np.ndarray
    n_hops: np.ndarray
    cot_ids: np.ndarray
    cot_mask: np.ndarray
    cot_loss_mask: np.ndarray


def collate(insts: list[dict], vocab: Vocab, max_steps: int,
            prompt_len: int | None = None, cot_len: int | None = None) -> Batch:
    """Tokenize + pad a batch. Pass fixed ``prompt_len``/``cot_len`` (global
    maxima) so every batch shares shapes and JAX compiles the model *once*
    instead of per batch — a large speedup."""
    r = [render(i, vocab) for i in insts]
    B = len(r)
    Sp = prompt_len if prompt_len is not None else max(len(x.prompt) for x in r)
    assert Sp >= max(len(x.prompt) for x in r), "prompt_len too small for batch"
    K = max_steps

    prompt_ids = np.full((B, Sp), PAD, np.int32)
    prompt_mask = np.zeros((B, Sp), np.float32)
    answer = np.zeros((B,), np.int32)
    path_targets = np.full((B, K), -1, np.int32)
    path_len = np.zeros((B,), np.int32)
    n_hops = np.zeros((B,), np.int32)

    # CoT sequence = prompt + steps + <ans> + answer + <eos>
    cot_seqs = []
    for x in r:
        seq = list(x.prompt) + list(x.steps) + [ANS, x.answer, EOS]
        loss_start = len(x.prompt)  # score steps + answer (not the prompt)
        cot_seqs.append((seq, loss_start))
    Sc = cot_len if cot_len is not None else max(len(s) for s, _ in cot_seqs)
    assert Sc >= max(len(s) for s, _ in cot_seqs), "cot_len too small for batch"
    cot_ids = np.full((B, Sc), PAD, np.int32)
    cot_mask = np.zeros((B, Sc), np.float32)
    cot_loss_mask = np.zeros((B, Sc), np.float32)

    for b, x in enumerate(r):
        # LEFT-pad the prompt so every prompt ends at column Sp-1; the latent
        # thought slots then follow uniformly for all examples (see latent.py).
        lp = len(x.prompt)
        prompt_ids[b, Sp - lp :] = x.prompt
        prompt_mask[b, Sp - lp :] = 1.0
        answer[b] = x.answer
        pt = x.path_targets[:K]
        path_targets[b, : len(pt)] = pt
        path_len[b] = min(len(x.path_targets), K)
        n_hops[b] = x.n_hops
        seq, loss_start = cot_seqs[b]
        cot_ids[b, : len(seq)] = seq
        cot_mask[b, : len(seq)] = 1.0
        cot_loss_mask[b, loss_start : len(seq)] = 1.0

    return Batch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        answer=answer,
        path_targets=path_targets,
        path_len=path_len,
        n_hops=n_hops,
        cot_ids=cot_ids,
        cot_mask=cot_mask,
        cot_loss_mask=cot_loss_mask,
    )


def iterate_batches(insts: list[dict], vocab: Vocab, batch_size: int,
                    max_steps: int, seed: int, shuffle: bool = True):
    """Deterministic batch iterator: order is a pure function of (seed, epoch)."""
    idx = np.arange(len(insts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(insts) - batch_size + 1, batch_size):
        chunk = [insts[j] for j in idx[start : start + batch_size]]
        yield collate(chunk, vocab, max_steps)
