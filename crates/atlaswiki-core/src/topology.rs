//! Pure-Rust Node2Vec graph embeddings and topological similarity engine.
//!
//! Provides biased second-order random walks (parameters p return, q in-out),
//! Skip-gram SGD with negative sampling (32-dimensional vector embeddings),
//! and fast cosine topological similarity queries in pure Rust (zero PyTorch, zero external C/C++ dependencies).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::graph::KnowledgeGraph;
use crate::storage::StorageEngine;

/// High-entropy, deterministic Pseudo-Random Number Generator (Xoshiro256++).
///
/// Implemented in pure Rust with zero external dependencies.
#[derive(Clone, Debug)]
pub struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    /// Create a new PRNG seeded from a 64-bit integer using SplitMix64 expansion.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_u64 = || {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next_u64(), next_u64(), next_u64(), next_u64()],
        }
    }

    /// Create a new PRNG seeded from the system clock entropy.
    pub fn from_entropy() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c49e6748fea9b);
        Self::seed_from_u64(seed)
    }

    /// Generate next pseudo-random 64-bit unsigned integer.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate random usize uniformly in 0..bound.
    #[inline]
    pub fn gen_range(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    /// Generate random f32 uniformly in [0.0, 1.0).
    #[inline]
    pub fn gen_f32(&mut self) -> f32 {
        let val = (self.next_u64() >> 40) as u32;
        val as f32 / 16777216.0
    }
}

/// Hyperparameters for Node2Vec random walks and Skip-gram SGD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2VecConfig {
    /// Return parameter p (likelihood of revisiting immediate predecessor).
    pub p: f32,
    /// In-out parameter q (likelihood of exploring outward vs staying local).
    pub q: f32,
    /// Embedding vector dimensionality (AtlasWiki default: 32).
    pub dimensions: usize,
    /// Length of each random walk path.
    pub walk_length: usize,
    /// Number of random walks generated per node.
    pub walks_per_node: usize,
    /// Skip-gram context window size.
    pub window_size: usize,
    /// Number of negative samples per positive center-context pair.
    pub negative_samples: usize,
    /// Total training epochs over generated walk corpora.
    pub epochs: usize,
    /// Initial learning rate for SGD.
    pub learning_rate: f32,
    /// Minimum learning rate after decay.
    pub min_learning_rate: f32,
    /// Optional deterministic seed for reproducible execution.
    pub seed: Option<u64>,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Self {
            p: 1.0,
            q: 1.0,
            dimensions: 32,
            walk_length: 20,
            walks_per_node: 10,
            window_size: 3,
            negative_samples: 5,
            epochs: 5,
            learning_rate: 0.025,
            min_learning_rate: 0.0001,
            seed: None,
        }
    }
}

/// Pure-Rust Node2Vec graph embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2Vec {
    config: Node2VecConfig,
    node_names: Vec<String>,
    name_to_index: HashMap<String, usize>,
    title_map: HashMap<String, usize>, // lowercase title -> index
    #[serde(skip)]
    neighbors: Vec<Vec<(usize, f32)>>, // neighbor index, edge weight
    #[serde(skip)]
    neighbor_sets: Vec<HashSet<usize>>, // for O(1) 2nd order connectivity check
    #[serde(skip)]
    neighbor_weights_sum: Vec<f32>,
    embeddings: Vec<Vec<f32>>,
}

/// Numerically stable sigmoid activation function.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z > 15.0 {
        1.0
    } else if z < -15.0 {
        0.0
    } else {
        1.0 / (1.0 + (-z).exp())
    }
}

impl Node2Vec {
    /// Create an empty Node2Vec instance with specified configuration.
    pub fn new(config: Node2VecConfig) -> Self {
        Self {
            config,
            node_names: Vec::new(),
            name_to_index: HashMap::new(),
            title_map: HashMap::new(),
            neighbors: Vec::new(),
            neighbor_sets: Vec::new(),
            neighbor_weights_sum: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    /// Construct Node2Vec model from an AtlasWiki `KnowledgeGraph`.
    pub fn from_knowledge_graph(kg: &KnowledgeGraph, config: Node2VecConfig) -> Self {
        let n = kg.node_count();
        let raw_graph = kg.raw_graph();

        let mut node_names = Vec::with_capacity(n);
        let mut name_to_index = HashMap::with_capacity(n);
        let mut title_map = HashMap::with_capacity(n);

        // Map petgraph NodeIndex (0..n) to sequential index
        for i in 0..n {
            let p_idx = petgraph::graph::NodeIndex::new(i);
            let title = match raw_graph.node_weight(p_idx) {
                Some(node) => node.title.clone(),
                None => format!("node_{}", i),
            };
            name_to_index.insert(title.clone(), i);
            title_map.insert(title.to_lowercase(), i);
            node_names.push(title);
        }

        // Aggregate undirected edges with weight accumulation
        let mut adj_map: Vec<HashMap<usize, f32>> = vec![HashMap::new(); n];
        for edge in raw_graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            if u < n && v < n && u != v {
                let w = edge.weight().weight.max(0.1);
                *adj_map[u].entry(v).or_insert(0.0) += w;
                *adj_map[v].entry(u).or_insert(0.0) += w;
            }
        }

        let mut neighbors = Vec::with_capacity(n);
        let mut neighbor_sets = Vec::with_capacity(n);
        let mut neighbor_weights_sum = Vec::with_capacity(n);

        for i in 0..n {
            let nbr_map = std::mem::take(&mut adj_map[i]);
            let mut nbr_vec: Vec<(usize, f32)> = nbr_map.into_iter().collect();
            nbr_vec.sort_by_key(|&(nbr, _)| nbr); // deterministic sorting

            let nbr_set: HashSet<usize> = nbr_vec.iter().map(|&(nbr, _)| nbr).collect();
            let sum_w: f32 = nbr_vec.iter().map(|&(_, w)| w).sum();

            neighbors.push(nbr_vec);
            neighbor_sets.push(nbr_set);
            neighbor_weights_sum.push(sum_w);
        }

        Self {
            config,
            node_names,
            name_to_index,
            title_map,
            neighbors,
            neighbor_sets,
            neighbor_weights_sum,
            embeddings: Vec::new(),
        }
    }

    /// Construct Node2Vec model from explicit node titles and weighted edge tuples (u, v, weight).
    pub fn from_edges(
        nodes: Vec<String>,
        edges: &[(usize, usize, f32)],
        config: Node2VecConfig,
    ) -> Self {
        let n = nodes.len();
        let mut name_to_index = HashMap::with_capacity(n);
        let mut title_map = HashMap::with_capacity(n);

        for (i, name) in nodes.iter().enumerate() {
            name_to_index.insert(name.clone(), i);
            title_map.insert(name.to_lowercase(), i);
        }

        let mut adj_map: Vec<HashMap<usize, f32>> = vec![HashMap::new(); n];
        for &(u, v, w) in edges {
            if u < n && v < n && u != v {
                let weight = w.max(0.1);
                *adj_map[u].entry(v).or_insert(0.0) += weight;
                *adj_map[v].entry(u).or_insert(0.0) += weight;
            }
        }

        let mut neighbors = Vec::with_capacity(n);
        let mut neighbor_sets = Vec::with_capacity(n);
        let mut neighbor_weights_sum = Vec::with_capacity(n);

        for i in 0..n {
            let nbr_map = std::mem::take(&mut adj_map[i]);
            let mut nbr_vec: Vec<(usize, f32)> = nbr_map.into_iter().collect();
            nbr_vec.sort_by_key(|&(nbr, _)| nbr);

            let nbr_set: HashSet<usize> = nbr_vec.iter().map(|&(nbr, _)| nbr).collect();
            let sum_w: f32 = nbr_vec.iter().map(|&(_, w)| w).sum();

            neighbors.push(nbr_vec);
            neighbor_sets.push(nbr_set);
            neighbor_weights_sum.push(sum_w);
        }

        Self {
            config,
            node_names: nodes,
            name_to_index,
            title_map,
            neighbors,
            neighbor_sets,
            neighbor_weights_sum,
            embeddings: Vec::new(),
        }
    }

    /// Construct Node2Vec model directly from pre-computed embeddings.
    pub fn from_embeddings(embeddings: Vec<(String, Vec<f32>)>, config: Node2VecConfig) -> Self {
        let n = embeddings.len();
        let mut node_names = Vec::with_capacity(n);
        let mut name_to_index = HashMap::with_capacity(n);
        let mut title_map = HashMap::with_capacity(n);
        let mut embs = Vec::with_capacity(n);

        for (i, (title, vec)) in embeddings.into_iter().enumerate() {
            name_to_index.insert(title.clone(), i);
            title_map.insert(title.to_lowercase(), i);
            node_names.push(title);
            embs.push(vec);
        }

        Self {
            config,
            node_names,
            name_to_index,
            title_map,
            neighbors: vec![Vec::new(); n],
            neighbor_sets: vec![HashSet::new(); n],
            neighbor_weights_sum: vec![0.0; n],
            embeddings: embs,
        }
    }

    /// Total number of notes indexed in the graph.
    pub fn node_count(&self) -> usize {
        self.node_names.len()
    }

    /// Reference to configuration hyperparameters.
    pub fn config(&self) -> &Node2VecConfig {
        &self.config
    }

    /// Names of all nodes indexed in this model.
    pub fn node_names(&self) -> &[String] {
        &self.node_names
    }

    /// Get learned embedding vector for a note title if available.
    pub fn get_embedding(&self, note: &str) -> Option<&[f32]> {
        let clean = note.trim().trim_start_matches("[[").trim_end_matches("]]");
        let idx = self.name_to_index.get(clean)
            .or_else(|| self.title_map.get(&clean.to_lowercase()))?;
        self.embeddings.get(*idx).map(|v| v.as_slice())
    }

    /// Return all note titles and their learned embedding vectors.
    pub fn all_embeddings(&self) -> Vec<(String, Vec<f32>)> {
        self.node_names
            .iter()
            .cloned()
            .zip(self.embeddings.iter().cloned())
            .collect()
    }

    /// Sample the first step from a starting node based purely on edge weights.
    fn sample_first_step(&self, node: usize, rng: &mut FastRng) -> usize {
        let nbrs = &self.neighbors[node];
        let total_w = self.neighbor_weights_sum[node];
        if total_w <= 0.0 || nbrs.len() == 1 {
            return nbrs[0].0;
        }

        let mut r = rng.gen_f32() * total_w;
        for &(nbr, w) in nbrs {
            r -= w;
            if r <= 0.0 {
                return nbr;
            }
        }
        nbrs.last().unwrap().0
    }

    /// Sample the next step using biased 2nd-order Node2Vec transition probabilities:
    /// - alpha(t, x) = 1/p   if d_tx == 0 (x == prev)
    /// - alpha(t, x) = 1.0   if d_tx == 1 (x is connected to prev)
    /// - alpha(t, x) = 1/q   if d_tx == 2 (x is not connected to prev)
    pub fn sample_second_order_step(&self, prev: usize, curr: usize, rng: &mut FastRng) -> usize {
        let nbrs = &self.neighbors[curr];
        if nbrs.is_empty() {
            return curr;
        }
        if nbrs.len() == 1 {
            return nbrs[0].0;
        }

        let prev_nbr_set = &self.neighbor_sets[prev];
        let inv_p = 1.0 / self.config.p.max(1e-5);
        let inv_q = 1.0 / self.config.q.max(1e-5);

        let mut total_w = 0.0f32;
        let mut weights = Vec::with_capacity(nbrs.len());

        for &(x, edge_w) in nbrs {
            let alpha = if x == prev {
                inv_p
            } else if prev_nbr_set.contains(&x) {
                1.0
            } else {
                inv_q
            };
            let w = alpha * edge_w;
            weights.push(w);
            total_w += w;
        }

        if total_w <= 0.0 {
            return nbrs[rng.gen_range(nbrs.len())].0;
        }

        let mut r = rng.gen_f32() * total_w;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return nbrs[i].0;
            }
        }
        nbrs.last().unwrap().0
    }

    /// Simulate a single Node2Vec biased second-order random walk starting at `start_node`.
    pub fn node2vec_walk(&self, start_node: usize, rng: &mut FastRng) -> Vec<usize> {
        let mut walk = Vec::with_capacity(self.config.walk_length);
        walk.push(start_node);

        if self.config.walk_length <= 1 {
            return walk;
        }

        let curr_nbrs = &self.neighbors[start_node];
        if curr_nbrs.is_empty() {
            return walk;
        }

        // First step (1st order transition)
        let first_step = self.sample_first_step(start_node, rng);
        walk.push(first_step);

        // Subsequent steps (2nd order biased walk using p and q)
        while walk.len() < self.config.walk_length {
            let curr = walk[walk.len() - 1];
            let prev = walk[walk.len() - 2];

            if self.neighbors[curr].is_empty() {
                break;
            }

            let next_node = self.sample_second_order_step(prev, curr, rng);
            walk.push(next_node);
        }

        walk
    }

    /// Generate all second-order random walks across all nodes in the knowledge graph.
    pub fn generate_walks(&self, rng: &mut FastRng) -> Vec<Vec<usize>> {
        let n = self.node_names.len();
        if n == 0 {
            return Vec::new();
        }

        let mut walks = Vec::with_capacity(n * self.config.walks_per_node);
        let mut nodes: Vec<usize> = (0..n).collect();

        for _ in 0..self.config.walks_per_node {
            // Fisher-Yates shuffle across nodes to randomize walk order
            for i in (1..n).rev() {
                let j = rng.gen_range(i + 1);
                nodes.swap(i, j);
            }

            for &start_node in &nodes {
                let walk = self.node2vec_walk(start_node, rng);
                if !walk.is_empty() {
                    walks.push(walk);
                }
            }
        }

        walks
    }

    /// Train 32-dimensional embedding vectors using pure-Rust Skip-gram SGD with negative sampling.
    pub fn train(&mut self) {
        let n = self.node_names.len();
        if n == 0 {
            self.embeddings.clear();
            return;
        }
        if n == 1 {
            let val = 1.0 / (self.config.dimensions as f32).sqrt();
            self.embeddings = vec![vec![val; self.config.dimensions]];
            return;
        }

        let mut rng = match self.config.seed {
            Some(s) => FastRng::seed_from_u64(s),
            None => FastRng::from_entropy(),
        };

        // 1. Generate second-order biased random walks
        let walks = self.generate_walks(&mut rng);
        if walks.is_empty() {
            return;
        }

        // 2. Count node occurrences for frequency-weighted negative sampling
        let mut node_counts = vec![0usize; n];
        for walk in &walks {
            for &node in walk {
                if node < n {
                    node_counts[node] += 1;
                }
            }
        }

        // 3. Construct unigram negative sampling table (scaled by degree / frequency ^ 0.75)
        let table_size = 100_000.max(n * 10);
        let mut neg_table = Vec::with_capacity(table_size);
        let mut powered_sum = 0.0f64;
        let powered_counts: Vec<f64> = node_counts
            .iter()
            .map(|&c| {
                let p = (c.max(1) as f64).powf(0.75);
                powered_sum += p;
                p
            })
            .collect();

        for (node, &p) in powered_counts.iter().enumerate() {
            let entries = ((p / powered_sum) * table_size as f64).round() as usize;
            for _ in 0..entries {
                neg_table.push(node);
            }
        }
        while neg_table.len() < table_size {
            neg_table.push(rng.gen_range(n));
        }

        // 4. Initialize embedding matrices W (input) and C (context)
        let d = self.config.dimensions;
        let scale = 0.5 / (d as f32).sqrt();
        let mut w_mat = vec![0.0f32; n * d];
        for i in 0..w_mat.len() {
            w_mat[i] = (rng.gen_f32() - 0.5) * 2.0 * scale;
        }
        let mut c_mat = vec![0.0f32; n * d]; // standard Word2Vec zero-initialized context vectors

        // 5. Compute total context pairs for linear learning rate decay
        let mut total_pairs = 0usize;
        for walk in &walks {
            let len = walk.len();
            for i in 0..len {
                let start = i.saturating_sub(self.config.window_size);
                let end = (i + self.config.window_size + 1).min(len);
                for j in start..end {
                    if i != j {
                        total_pairs += 1;
                    }
                }
            }
        }
        let total_steps = (total_pairs * self.config.epochs).max(1);
        let mut current_step = 0usize;
        let initial_lr = self.config.learning_rate;
        let min_lr = self.config.min_learning_rate;

        let mut grad_u = vec![0.0f32; d];

        // 6. SGD training loop
        for _epoch in 0..self.config.epochs {
            for walk in &walks {
                let len = walk.len();
                for i in 0..len {
                    let u = walk[i];
                    if u >= n {
                        continue;
                    }

                    let start = i.saturating_sub(self.config.window_size);
                    let end = (i + self.config.window_size + 1).min(len);

                    for j in start..end {
                        if i == j {
                            continue;
                        }
                        let v = walk[j];
                        if v >= n {
                            continue;
                        }

                        // Linear learning rate decay
                        let progress = current_step as f32 / total_steps as f32;
                        let lr = (initial_lr * (1.0 - progress)).max(min_lr);
                        current_step += 1;

                        grad_u.fill(0.0);

                        let u_offset = u * d;
                        let v_offset = v * d;

                        // Positive pair (u, v) with label = 1
                        let mut dot = 0.0f32;
                        for k in 0..d {
                            dot += w_mat[u_offset + k] * c_mat[v_offset + k];
                        }
                        let sig = sigmoid(dot);
                        let g_pos = lr * (1.0 - sig);

                        for k in 0..d {
                            grad_u[k] += g_pos * c_mat[v_offset + k];
                            c_mat[v_offset + k] += g_pos * w_mat[u_offset + k];
                        }

                        // Negative samples (u, neg) with label = 0
                        for _ in 0..self.config.negative_samples {
                            let neg_node = neg_table[rng.gen_range(neg_table.len())];
                            if neg_node == u || neg_node == v {
                                continue;
                            }

                            let neg_offset = neg_node * d;
                            let mut neg_dot = 0.0f32;
                            for k in 0..d {
                                neg_dot += w_mat[u_offset + k] * c_mat[neg_offset + k];
                            }
                            let neg_sig = sigmoid(neg_dot);
                            let g_neg = lr * (0.0 - neg_sig);

                            for k in 0..d {
                                grad_u[k] += g_neg * c_mat[neg_offset + k];
                                c_mat[neg_offset + k] += g_neg * w_mat[u_offset + k];
                            }
                        }

                        // Gradient update for W[u]
                        for k in 0..d {
                            w_mat[u_offset + k] += grad_u[k];
                        }
                    }
                }
            }
        }

        // 7. L2 normalization for fast cosine similarity via dot product
        self.embeddings = Vec::with_capacity(n);
        for i in 0..n {
            let u_offset = i * d;
            let mut row = Vec::with_capacity(d);
            let mut norm_sq = 0.0f32;
            for k in 0..d {
                let val = w_mat[u_offset + k];
                row.push(val);
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-10 {
                for val in &mut row {
                    *val /= norm;
                }
            }
            self.embeddings.push(row);
        }
    }

    /// Retrieve the top `k` notes most topologically similar to `note` based on cosine similarity.
    pub fn most_topologically_similar(&self, note: &str, k: usize) -> Vec<(String, f32)> {
        if k == 0 || self.embeddings.is_empty() {
            return Vec::new();
        }

        let clean_note = note.trim().trim_start_matches("[[").trim_end_matches("]]");
        let target_idx = match self
            .name_to_index
            .get(clean_note)
            .or_else(|| self.title_map.get(&clean_note.to_lowercase()))
        {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let target_vec = match self.embeddings.get(target_idx) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let d = self.config.dimensions;

        let mut scores: Vec<(String, f32)> = Vec::with_capacity(self.node_names.len());

        for (i, name) in self.node_names.iter().enumerate() {
            if i == target_idx {
                continue; // Exclude target note itself
            }
            if let Some(vec) = self.embeddings.get(i) {
                let mut sim = 0.0f32;
                for c in 0..d {
                    sim += target_vec[c] * vec[c];
                }
                scores.push((name.clone(), sim));
            }
        }

        // Sort descending by similarity score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Persist learned node embeddings into SQLite storage.
    pub fn save_to_storage(&self, storage: &StorageEngine) -> Result<()> {
        let pairs = self.all_embeddings();
        storage.save_node_embeddings(&pairs, "node2vec")
    }

    /// Load node embeddings from SQLite storage into a new `Node2Vec` model.
    pub fn load_from_storage(storage: &StorageEngine) -> Result<Self> {
        let embeddings = storage.load_node_embeddings("node2vec")?;
        Ok(Self::from_embeddings(embeddings, Node2VecConfig::default()))
    }

    /// Save embeddings and metadata as a JSON file.
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).context("Failed to create JSON embeddings file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load embeddings and metadata from a JSON file.
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).context("Failed to open JSON embeddings file")?;
        let reader = BufReader::new(file);
        let model: Self = serde_json::from_reader(reader)?;
        Ok(model)
    }
}
