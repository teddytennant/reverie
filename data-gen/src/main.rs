//! Deterministic generator of ProsQA-style latent-planning tasks.
//!
//! Each instance is a small DAG over fictional "concept" entities with an
//! `is-a` hierarchy. Given the facts, decide which of two candidate concepts
//! the source concept *is*, i.e. which of the two it can reach. Dead-end
//! branches hang off the true path, so answering takes a search and not a
//! lookup.
//!
//! Rust owns the abstract problem (graph plus gold path); the Python side
//! renders it to token text. Output is JSONL, one instance per line.

use std::io::{BufWriter, Write};

/// SplitMix64. Hand-rolled rather than pulled from `rand` so the stream is
/// pinned to this file and a dependency bump can't quietly move the dataset.
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

const CONS: &[u8] = b"bcdfghjklmnpqrstvwxz";
const VOWELS: &[u8] = b"aeiou";

/// Pronounceable fictional name for a node index.
///
/// Must agree with `concept_name` in reverie/data.py: the Python side builds
/// its vocabulary from that function and looks entities up by name, so if the
/// two ever drift apart every entity gets silently mislabeled.
///
/// The names are made up on purpose. Real words would let a model answer from
/// world knowledge instead of searching the graph.
fn concept_name(mut idx: usize) -> String {
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

// ---- graph construction

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
    connect: usize,  // decoy->source cross-edges: weakly-connect the graph so a
                     // component-membership check can't tell the two candidates
                     // apart. Labels are unaffected; see gen_instance.
    out: Option<String>,
}

/// Directed reachability. Every emitted label is checked with this instead of
/// being taken on faith from the construction.
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
    // Two disjoint id ranges keep the components apart by construction: the
    // source component (backbone plus traps) and the decoy component holding
    // the unreachable candidate. Only the --connect edges cross between them.
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

    // Backbone: the true path from source to t_pos. Ids increase along it.
    let backbone: Vec<usize> = (0..path_len).map(|i| src_base + i).collect();
    for i in 0..cfg.hops {
        add(&mut edges, &mut adj, backbone[i], backbone[i + 1]);
    }
    let source = backbone[0];
    let t_pos = backbone[cfg.hops];

    // Dead-end chains hanging off each backbone node. A trap id is always
    // greater than its parent's, which keeps the graph acyclic and means no
    // trap can lead back into the backbone or on to t_pos.
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

    // t_neg gets its own is-a chain so it reads like a real concept. Nothing
    // in here is reachable from source.
    let decoy: Vec<usize> = (0..n_decoy).map(|i| decoy_base + i).collect();
    for i in 0..n_decoy - 1 {
        add(&mut edges, &mut adj, decoy[i], decoy[i + 1]);
    }
    let t_neg = decoy[rng.below(n_decoy)];

    // Cross-edges, decoy->source. They point into the source component and
    // never out of it, so what the source can reach doesn't change and t_neg
    // stays unreachable. What does change: the graph is now one weakly
    // connected blob, so "which component is the source in" stops separating
    // the two candidates and only directed reachability answers it.
    for _ in 0..cfg.connect {
        let u = decoy_base + rng.below(n_decoy);      // a decoy-component node
        let v = src_base + rng.below(src_count);      // a source-component node
        if u != v {
            add(&mut edges, &mut adj, u, v);
        }
    }

    // Check both labels, then shuffle ids so position gives nothing away.
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

// ---- CLI

fn parse_args() -> Config {
    let mut cfg = Config {
        n: 10_000,
        seed: 0,
        hops: 4,
        branch: 2,
        trap_depth: 2,
        connect: 0,
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
            "--connect" => cfg.connect = val().parse().unwrap(),
            "--out" => cfg.out = Some(val()),
            "--help" | "-h" => {
                eprintln!("reverie-datagen --n N --seed S --hops H --branch B --trap-depth D --connect C --out FILE");
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
        // Each instance draws from its own sub-stream, so instance 8471 can be
        // regenerated on its own without replaying the 8471 before it.
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
        Config { n: 0, seed: 0, hops: 4, branch: 2, trap_depth: 2, connect: 0, out: None }
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

    #[test]
    fn cross_edges_preserve_labels() {
        // The cross-edges make the graph weakly connected. The source still
        // must not reach the decoy.
        let mut cfg = test_cfg();
        cfg.connect = 4;
        for id in 0..300 {
            let mut rng = Rng::new(9 ^ 0x9E3779B97F4A7C15u64.wrapping_mul(id as u64 + 1));
            let inst = gen_instance(id, &mut rng, &cfg);
            let mut adj = vec![Vec::new(); inst.n_entities];
            for &(a, b) in &inst.edges {
                adj[a].push(b);
            }
            let ans = inst.answer;
            let decoy = if inst.candidates[0] == ans { inst.candidates[1] } else { inst.candidates[0] };
            assert!(reachable(&adj, inst.source, ans));
            assert!(!reachable(&adj, inst.source, decoy), "decoy must stay unreachable with cross-edges");
        }
    }
}
