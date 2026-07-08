//! Deterministic generator of ProsQA-style latent-planning tasks.
//!
//! Each instance is a small DAG over fictional "concept" entities with an
//! `is-a` hierarchy. The task: given the facts, decide which of two candidate
//! concepts the source concept *is* (i.e. which target is reachable). Distractor
//! branches off the true path turn this into a graph-search / planning problem —
//! exactly the regime where reasoning in latent space (implicit BFS) is claimed
//! to beat left-to-right chain-of-thought.
//!
//! Rust owns the *abstract* problem (graph + gold path); the Python side renders
//! it to token text. Output is JSONL (one instance per line). Zero dependencies:
//! deterministic PRNG and JSON are hand-rolled for an instant, offline build.

use std::io::{BufWriter, Write};

// ---------------------------------------------------------------- PRNG

/// SplitMix64 — tiny, fast, deterministic. Same seed => same dataset, forever.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------- names

const CONS: &[u8] = b"bcdfghjklmnpqrstvwxz";
const VOWELS: &[u8] = b"aeiou";

/// Deterministic, unique, pronounceable fictional name for a node index.
/// Fictional names stop the model from leaning on real-world semantics — the
/// only signal is the graph structure it must search.
fn concept_name(mut idx: usize) -> String {
    // Base-(20*5) syllables => plenty of unique short names.
    let mut s = String::new();
    loop {
        let c = CONS[idx % CONS.len()] as char;
        idx /= CONS.len();
        let v = VOWELS[idx % VOWELS.len()] as char;
        idx /= VOWELS.len();
        s.push(c);
        s.push(v);
        if idx == 0 {
            break;
        }
    }
    s
}

// ---------------------------------------------------------------- graph

struct Instance {
    id: usize,
    n_entities: usize,
    edges: Vec<(usize, usize)>, // directed: "every a is a b"
    source: usize,
    candidates: [usize; 2],
    answer: usize,
    gold_path: Vec<usize>,
    n_hops: usize,
    n_distractors: usize,
}

struct Config {
    n: usize,
    seed: u64,
    hops: usize,     // length of the true reasoning path
    branch: usize,   // distractor edges per path node
    trap_depth: usize, // how deep distractor chains go
    out: Option<String>,
}

/// Reachability check used to *verify* every generated label — we never trust
/// construction alone.
fn reachable(adj: &[Vec<usize>], src: usize, dst: usize) -> bool {
    let mut seen = vec![false; adj.len()];
    let mut stack = vec![src];
    seen[src] = true;
    while let Some(u) = stack.pop() {
        if u == dst {
            return true;
        }
        for &w in &adj[u] {
            if !seen[w] {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    false
}

fn gen_instance(id: usize, rng: &mut Rng, cfg: &Config) -> Instance {
    // Two disjoint node-id ranges keep the DAG's components separate by
    // construction: the source component (backbone + traps) and the decoy
    // component (holds the unreachable candidate). Edges never cross ranges.
    let path_len = cfg.hops + 1;
    let n_traps = cfg.branch * cfg.hops * cfg.trap_depth + cfg.branch;
    let n_decoy = cfg.hops + 2; // decoy gets its own plausible hierarchy

    let src_base = 0usize;
    let src_count = path_len + n_traps;
    let decoy_base = src_count;
    let total = src_count + n_decoy;

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total];
    let add = |edges: &mut Vec<(usize, usize)>, adj: &mut Vec<Vec<usize>>, a: usize, b: usize| {
        edges.push((a, b));
        adj[a].push(b);
    };

    // --- backbone: source -> ... -> t_pos, strictly increasing id => acyclic
    let backbone: Vec<usize> = (0..path_len).map(|i| src_base + i).collect();
    for i in 0..cfg.hops {
        add(&mut edges, &mut adj, backbone[i], backbone[i + 1]);
    }
    let source = backbone[0];
    let t_pos = backbone[cfg.hops];

    // --- distractor traps: branch off each path node into dead-end chains.
    // Trap ids are all > their parents so the graph stays a DAG and traps can
    // never loop back to the backbone or reach t_pos.
    let mut next_trap = src_base + path_len;
    let mut n_distractors = 0;
    for i in 0..cfg.hops {
        for _ in 0..cfg.branch {
            if next_trap >= decoy_base {
                break;
            }
            let mut prev = backbone[i];
            for _ in 0..cfg.trap_depth {
                if next_trap >= decoy_base {
                    break;
                }
                let t = next_trap;
                next_trap += 1;
                add(&mut edges, &mut adj, prev, t);
                n_distractors += 1;
                prev = t;
            }
        }
    }

    // --- decoy component: give t_neg its own is-a chain so it reads as a real
    // concept, but nothing here is reachable from source.
    let decoy: Vec<usize> = (0..n_decoy).map(|i| decoy_base + i).collect();
    for i in 0..n_decoy - 1 {
        add(&mut edges, &mut adj, decoy[i], decoy[i + 1]);
    }
    let t_neg = decoy[rng.below(n_decoy)];

    // --- verify labels, then shuffle entity ids so position leaks nothing.
    assert!(reachable(&adj, source, t_pos), "t_pos must be reachable");
    assert!(!reachable(&adj, source, t_neg), "t_neg must be unreachable");

    let mut perm: Vec<usize> = (0..total).collect();
    rng.shuffle(&mut perm);
    // inverse map old->new
    let mut relabel = vec![0usize; total];
    for (new_id, &old_id) in perm.iter().enumerate() {
        relabel[old_id] = new_id;
    }
    let mut r_edges: Vec<(usize, usize)> =
        edges.iter().map(|&(a, b)| (relabel[a], relabel[b])).collect();
    rng.shuffle(&mut r_edges);

    let gold_path: Vec<usize> = backbone.iter().map(|&n| relabel[n]).collect();
    let mut candidates = [relabel[t_pos], relabel[t_neg]];
    if rng.below(2) == 1 {
        candidates.swap(0, 1);
    }

    Instance {
        id,
        n_entities: total,
        edges: r_edges,
        source: relabel[source],
        candidates,
        answer: relabel[t_pos],
        gold_path,
        n_hops: cfg.hops,
        n_distractors,
    }
}

// ---------------------------------------------------------------- JSON

fn write_instance<W: Write>(w: &mut W, inst: &Instance) -> std::io::Result<()> {
    write!(w, "{{\"id\":{},", inst.id)?;
    write!(w, "\"n_entities\":{},", inst.n_entities)?;
    write!(w, "\"entities\":[")?;
    for i in 0..inst.n_entities {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "\"{}\"", concept_name(i))?;
    }
    write!(w, "],\"edges\":[")?;
    for (i, &(a, b)) in inst.edges.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "[{},{}]", a, b)?;
    }
    write!(w, "],\"source\":{},", inst.source)?;
    write!(w, "\"candidates\":[{},{}],", inst.candidates[0], inst.candidates[1])?;
    write!(w, "\"answer\":{},", inst.answer)?;
    write!(w, "\"gold_path\":[")?;
    for (i, &n) in inst.gold_path.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{}", n)?;
    }
    write!(w, "],\"n_hops\":{},", inst.n_hops)?;
    writeln!(w, "\"n_distractors\":{}}}", inst.n_distractors)?;
    Ok(())
}

// ---------------------------------------------------------------- CLI

fn parse_args() -> Config {
    let mut cfg = Config {
        n: 10_000,
        seed: 0,
        hops: 4,
        branch: 2,
        trap_depth: 2,
        out: None,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let key = args[i].as_str();
        let val = || args.get(i + 1).expect("missing value").clone();
        match key {
            "--n" => cfg.n = val().parse().unwrap(),
            "--seed" => cfg.seed = val().parse().unwrap(),
            "--hops" => cfg.hops = val().parse().unwrap(),
            "--branch" => cfg.branch = val().parse().unwrap(),
            "--trap-depth" => cfg.trap_depth = val().parse().unwrap(),
            "--out" => cfg.out = Some(val()),
            "--help" | "-h" => {
                eprintln!("reverie-datagen --n N --seed S --hops H --branch B --trap-depth D --out FILE");
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 2;
    }
    cfg
}

fn main() -> std::io::Result<()> {
    let cfg = parse_args();

    let sink: Box<dyn Write> = match &cfg.out {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };
    let mut w = BufWriter::new(sink);

    for id in 0..cfg.n {
        // Per-example sub-stream: any instance is regenerable in isolation and
        // generation is embarrassingly parallel, while staying bit-reproducible.
        let mut rng_i = Rng::new(cfg.seed ^ 0x9E3779B97F4A7C15u64.wrapping_mul(id as u64 + 1));
        let inst = gen_instance(id, &mut rng_i, &cfg);
        write_instance(&mut w, &inst)?;
    }
    w.flush()?;
    eprintln!(
        "generated {} instances (seed={}, hops={}, branch={}, trap_depth={})",
        cfg.n, cfg.seed, cfg.hops, cfg.branch, cfg.trap_depth
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> Config {
        Config { n: 0, seed: 0, hops: 4, branch: 2, trap_depth: 2, out: None }
    }

    fn instance_bytes(seed: u64, id: usize, cfg: &Config) -> Vec<u8> {
        let mut rng = Rng::new(seed ^ 0x9E3779B97F4A7C15u64.wrapping_mul(id as u64 + 1));
        let inst = gen_instance(id, &mut rng, cfg);
        let mut buf = Vec::new();
        write_instance(&mut buf, &inst).unwrap();
        buf
    }

    #[test]
    fn determinism_same_seed_same_bytes() {
        let cfg = test_cfg();
        for id in 0..50 {
            assert_eq!(instance_bytes(7, id, &cfg), instance_bytes(7, id, &cfg),
                       "instance {id} must be byte-reproducible");
        }
    }

    #[test]
    fn different_seed_differs() {
        let cfg = test_cfg();
        assert_ne!(instance_bytes(1, 0, &cfg), instance_bytes(2, 0, &cfg));
    }

    #[test]
    fn labels_are_bfs_verified() {
        let cfg = test_cfg();
        for id in 0..300 {
            let mut rng = Rng::new(3 ^ 0x9E3779B97F4A7C15u64.wrapping_mul(id as u64 + 1));
            let inst = gen_instance(id, &mut rng, &cfg);
            let mut adj = vec![Vec::new(); inst.n_entities];
            for &(a, b) in &inst.edges {
                adj[a].push(b);
            }
            let ans = inst.answer;
            let decoy = if inst.candidates[0] == ans { inst.candidates[1] } else { inst.candidates[0] };
            assert!(reachable(&adj, inst.source, ans), "answer must be reachable from source");
            assert!(!reachable(&adj, inst.source, decoy), "decoy must NOT be reachable");
            assert_eq!(*inst.gold_path.first().unwrap(), inst.source);
            assert_eq!(*inst.gold_path.last().unwrap(), ans);
        }
    }
}
