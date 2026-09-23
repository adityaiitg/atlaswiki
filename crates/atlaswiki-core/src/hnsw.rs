//! Embedded Hierarchical Navigable Small World (HNSW) and SQ8 Scalar Quantization
//! for AtlasWiki dense vector retrieval.
//!
//! Provides:
//! - `Sq8Vector`: Int8 scalar quantization with dynamic per-vector scaling,
//!   cosine similarity calculation, and asymmetric similarity querying.
//! - `HnswIndex`: Fast sub-millisecond approximate nearest neighbor (ANN) search
//!   over cosine distance metric with configurable `M`, `ef_construction`, and `ef_search`.
//! - `BruteForceScan`: Exact linear scan baseline for recall and precision benchmarking.
//! - Evaluation suite: Recall@10, Recall@50, latency percentiles, and synthetic vector generation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Fast deterministic pseudo-random number generator (XorShift64Star)
/// used for reproducible synthetic benchmarks and level generation.
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed
            },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform for generating standard normal random variables N(0, 1).
    pub fn next_gaussian(&mut self) -> (f32, f32) {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
    }
}

/// SQ8 (int8) Scalar Quantized Vector with dynamic per-vector scaling.
///
/// Encodes a 384-dimensional unit vector into 384 signed 8-bit integers (`i8`)
/// plus scale metadata, achieving a 4x reduction in memory footprint (1536 bytes -> 384 bytes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sq8Vector {
    pub data: Vec<i8>,
    pub scale: f32,
    pub inv_norm: f32,
}

impl Sq8Vector {
    /// Quantize an L2-normalized f32 vector into SQ8 format.
    pub fn encode(raw: &[f32]) -> Self {
        let max_abs = raw.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = if max_abs > 1e-12 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };

        let inv_scale = 1.0 / scale;
        let mut data = Vec::with_capacity(raw.len());
        let mut norm_sq = 0i64;

        for &val in raw {
            let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            norm_sq += (q as i64) * (q as i64);
            data.push(q);
        }

        let inv_norm = if norm_sq > 0 {
            1.0 / (norm_sq as f64).sqrt() as f32
        } else {
            1.0
        };

        Self {
            data,
            scale,
            inv_norm,
        }
    }

    /// Dequantize back into an L2-normalized f32 vector.
    pub fn decode(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.data.len());
        for &q in &self.data {
            out.push((q as f32) * self.inv_norm);
        }
        out
    }

    /// Compute symmetric cosine similarity between two SQ8 vectors using fast integer multiplication.
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.data.len(), other.data.len());
        let mut dot = 0i32;
        let n = self.data.len();
        for i in 0..n {
            dot += (self.data[i] as i32) * (other.data[i] as i32);
        }
        ((dot as f32) * self.inv_norm * other.inv_norm).clamp(-1.0, 1.0)
    }

    /// Compute asymmetric cosine similarity between an unquantized f32 query and this SQ8 vector.
    #[inline]
    pub fn cosine_similarity_f32(&self, query_f32: &[f32]) -> f32 {
        debug_assert_eq!(self.data.len(), query_f32.len());
        let mut dot = 0.0f32;
        for (&q, &d) in query_f32.iter().zip(self.data.iter()) {
            dot += q * (d as f32);
        }
        (dot * self.inv_norm).clamp(-1.0, 1.0)
    }

    /// Cosine distance = 1.0 - cosine_similarity. Range [0.0, 2.0].
    #[inline]
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        (1.0 - self.cosine_similarity(other)).max(0.0)
    }

    /// Asymmetric cosine distance between f32 query and SQ8 vector. Range [0.0, 2.0].
    #[inline]
    pub fn cosine_distance_f32(&self, query_f32: &[f32]) -> f32 {
        (1.0 - self.cosine_similarity_f32(query_f32)).max(0.0)
    }

    /// Raw heap byte footprint of this vector.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<Self>()
    }
}

/// Distance-node pair used in priority queues.
#[derive(Copy, Clone, PartialEq)]
struct DistNode {
    dist: f32,
    node: usize,
}

impl Eq for DistNode {}

impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration parameters for HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum outgoing connections per element at layers > 0.
    pub m: usize,
    /// Maximum outgoing connections per element at layer 0 (typically 2 * M).
    pub m0: usize,
    /// Size of dynamic candidate list during graph construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during query evaluation.
    pub ef_search: usize,
    /// Normalization factor for layer assignment: 1.0 / ln(M).
    pub ml: f64,
    /// Maximum number of hierarchical layers.
    pub max_layers: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 64,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
            max_layers: 16,
        }
    }
}

/// An individual node in the HNSW hierarchy.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HnswNode {
    id: usize,
    level: usize,
    /// Outgoing neighbor lists per layer: neighbors[level] is a list of node IDs.
    neighbors: Vec<Vec<usize>>,
}

/// Reusable visited set tracking for zero-allocation searches.
struct VisitedTracker {
    visited: Vec<u32>,
    epoch: u32,
}

impl VisitedTracker {
    fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; capacity],
            epoch: 1,
        }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.visited.len() < capacity {
            self.visited.resize(capacity, 0);
        }
    }

    fn advance(&mut self) {
        if self.epoch == u32::MAX {
            self.visited.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    #[inline]
    fn is_visited(&self, node: usize) -> bool {
        self.visited.get(node).copied().unwrap_or(0) == self.epoch
    }

    #[inline]
    fn mark_visited(&mut self, node: usize) {
        if node < self.visited.len() {
            self.visited[node] = self.epoch;
        }
    }
}

/// Embedded Hierarchical Navigable Small World (HNSW) vector search index.
pub struct HnswIndex {
    config: HnswConfig,
    vectors: Vec<Sq8Vector>,
    raw_vectors: Option<Vec<Vec<f32>>>,
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_level: usize,
    rng: FastRng,
}

impl HnswIndex {
    /// Create a new empty HNSW index with the specified configuration.
    pub fn new(config: HnswConfig) -> Self {
        let rng = FastRng::new(42);
        Self {
            config,
            vectors: Vec::new(),
            raw_vectors: None,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            rng,
        }
    }

    /// Create with default recommended configuration for 384-dimensional embeddings.
    pub fn with_default_config() -> Self {
        Self::new(HnswConfig::default())
    }

    /// Enable keeping raw FP32 vectors for hybrid asymmetric or exact re-ranking.
    pub fn with_raw_vectors(mut self, enable: bool) -> Self {
        if enable && self.raw_vectors.is_none() {
            self.raw_vectors = Some(Vec::new());
        } else if !enable {
            self.raw_vectors = None;
        }
        self
    }

    /// Total number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get current configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Total memory footprint in bytes (vectors + graph edges + node metadata).
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for v in &self.vectors {
            total += v.memory_bytes();
        }
        if let Some(raws) = &self.raw_vectors {
            for r in raws {
                total += r.len() * 4 + std::mem::size_of::<Vec<f32>>();
            }
        }
        for node in &self.nodes {
            total += std::mem::size_of::<HnswNode>();
            for layer in &node.neighbors {
                total += layer.len() * std::mem::size_of::<usize>() + std::mem::size_of::<Vec<usize>>();
            }
        }
        total
    }

    /// Assign a random level based on exponential decay with factor mL.
    fn sample_level(&mut self) -> usize {
        let r = self.rng.next_f64().max(1e-15);
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(self.config.max_layers - 1)
    }

    /// Insert an unquantized f32 vector into the index.
    pub fn insert(&mut self, id: usize, vector: &[f32]) {
        let sq8 = Sq8Vector::encode(vector);
        let raw = if self.raw_vectors.is_some() {
            Some(vector.to_vec())
        } else {
            None
        };
        self.insert_sq8(id, sq8, raw);
    }

    /// Insert a pre-quantized SQ8 vector into the index.
    pub fn insert_sq8(&mut self, id: usize, sq8: Sq8Vector, raw: Option<Vec<f32>>) {
        let level = self.sample_level();

        let mut node = HnswNode {
            id,
            level,
            neighbors: vec![Vec::new(); level + 1],
        };

        let internal_idx = self.nodes.len();
        debug_assert_eq!(internal_idx, id);

        self.vectors.push(sq8);
        if let Some(raw_vecs) = &mut self.raw_vectors {
            raw_vecs.push(raw.unwrap_or_default());
        }

        if let Some(entry_point) = self.entry_point {
            let mut curr_ep = entry_point;
            let query_vec = self.vectors[internal_idx].clone();
            let mut curr_dist = self.vectors[curr_ep].cosine_distance(&query_vec);

            // 1. Greedy descent down to node's level + 1
            if self.max_level > level {
                for lc in (level + 1..=self.max_level).rev() {
                    loop {
                        let mut changed = false;
                        for &neighbor in &self.nodes[curr_ep].neighbors[lc] {
                            let d = self.vectors[neighbor].cosine_distance(&query_vec);
                            if d < curr_dist {
                                curr_dist = d;
                                curr_ep = neighbor;
                                changed = true;
                            }
                        }
                        if !changed {
                            break;
                        }
                    }
                }
            }

            // 2. Multi-layer beam search and neighbor attachment from min(level, max_level) down to 0
            let top_level = level.min(self.max_level);
            let mut tracker = VisitedTracker::new(self.vectors.len());

            for lc in (0..=top_level).rev() {
                tracker.advance();
                let candidates = self.search_layer_internal(
                    &query_vec,
                    &[curr_ep],
                    self.config.ef_construction,
                    lc,
                    &mut tracker,
                );

                let m_max = if lc == 0 {
                    self.config.m0
                } else {
                    self.config.m
                };

                let selected = Self::select_neighbors(&candidates, m_max);
                node.neighbors[lc] = selected.clone();

                // Connect bidirectionally
                for &neighbor in &selected {
                    self.nodes[neighbor].neighbors[lc].push(internal_idx);
                    if self.nodes[neighbor].neighbors[lc].len() > m_max {
                        self.shrink_neighbors(neighbor, lc, m_max);
                    }
                }

                if let Some(&(_closest_d, closest_id)) = candidates.first() {
                    curr_ep = closest_id;
                }
            }

            if level > self.max_level {
                self.max_level = level;
                self.entry_point = Some(internal_idx);
            }
        } else {
            self.entry_point = Some(internal_idx);
            self.max_level = level;
        }

        self.nodes.push(node);
    }

    /// Simple neighbor selection: takes up to `m_max` nearest neighbors from candidates.
    fn select_neighbors(candidates: &[(f32, usize)], m_max: usize) -> Vec<usize> {
        candidates.iter().take(m_max).map(|&(_, id)| id).collect()
    }

    /// Prune outgoing edges of a node at a given layer to `m_max` closest.
    fn shrink_neighbors(&mut self, node: usize, level: usize, m_max: usize) {
        let base_vec = &self.vectors[node];
        let mut scored: Vec<(f32, usize)> = self.nodes[node].neighbors[level]
            .iter()
            .map(|&n| (base_vec.cosine_distance(&self.vectors[n]), n))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.nodes[node].neighbors[level] = scored.into_iter().take(m_max).map(|(_, id)| id).collect();
    }

    /// Internal layer search exploring nearest neighbors using Min-Max Heaps.
    fn search_layer_internal(
        &self,
        query: &Sq8Vector,
        entry_points: &[usize],
        ef: usize,
        level: usize,
        tracker: &mut VisitedTracker,
    ) -> Vec<(f32, usize)> {
        tracker.ensure_capacity(self.vectors.len());

        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::new();
        let mut w: BinaryHeap<DistNode> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.vectors[ep].cosine_distance(query);
            tracker.mark_visited(ep);
            candidates.push(Reverse(DistNode { dist: d, node: ep }));
            w.push(DistNode { dist: d, node: ep });
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let farthest = w.peek().unwrap();
            if c.dist > farthest.dist {
                break;
            }

            for &neighbor in &self.nodes[c.node].neighbors[level] {
                if !tracker.is_visited(neighbor) {
                    tracker.mark_visited(neighbor);
                    let d = self.vectors[neighbor].cosine_distance(query);
                    let farthest = w.peek().unwrap();

                    if w.len() < ef || d < farthest.dist {
                        candidates.push(Reverse(DistNode { dist: d, node: neighbor }));
                        w.push(DistNode { dist: d, node: neighbor });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<(f32, usize)> = w.into_iter().map(|dn| (dn.dist, dn.node)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search layer for asymmetric unquantized f32 query.
    fn search_layer_asymmetric(
        &self,
        query_f32: &[f32],
        entry_points: &[usize],
        ef: usize,
        level: usize,
        tracker: &mut VisitedTracker,
    ) -> Vec<(f32, usize)> {
        tracker.ensure_capacity(self.vectors.len());

        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::new();
        let mut w: BinaryHeap<DistNode> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.vectors[ep].cosine_distance_f32(query_f32);
            tracker.mark_visited(ep);
            candidates.push(Reverse(DistNode { dist: d, node: ep }));
            w.push(DistNode { dist: d, node: ep });
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let farthest = w.peek().unwrap();
            if c.dist > farthest.dist {
                break;
            }

            for &neighbor in &self.nodes[c.node].neighbors[level] {
                if !tracker.is_visited(neighbor) {
                    tracker.mark_visited(neighbor);
                    let d = self.vectors[neighbor].cosine_distance_f32(query_f32);
                    let farthest = w.peek().unwrap();

                    if w.len() < ef || d < farthest.dist {
                        candidates.push(Reverse(DistNode { dist: d, node: neighbor }));
                        w.push(DistNode { dist: d, node: neighbor });
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<(f32, usize)> = w.into_iter().map(|dn| (dn.dist, dn.node)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Execute approximate nearest neighbor (ANN) search for an unquantized f32 query.
    ///
    /// Uses asymmetric cosine distance (query in FP32, database nodes in SQ8)
    /// to maximize precision without dequantizing the database.
    ///
    /// Returns a list of `(node_id, cosine_similarity)` sorted descending by similarity.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        if self.is_empty() {
            return Vec::new();
        }

        let entry_point = self.entry_point.expect("HNSW must have entry point if non-empty");
        let mut curr_ep = entry_point;
        let mut curr_dist = self.vectors[curr_ep].cosine_distance_f32(query);

        // 1. Greedy descent down to layer 1
        for lc in (1..=self.max_level).rev() {
            loop {
                let mut changed = false;
                for &neighbor in &self.nodes[curr_ep].neighbors[lc] {
                    let d = self.vectors[neighbor].cosine_distance_f32(query);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // 2. Beam search at layer 0
        let ef = ef_search.max(k);
        let mut tracker = VisitedTracker::new(self.vectors.len());
        tracker.advance();

        let top_candidates = self.search_layer_asymmetric(query, &[curr_ep], ef, 0, &mut tracker);

        // 3. Convert distance back to cosine similarity and truncate to top-k
        top_candidates
            .into_iter()
            .take(k)
            .map(|(dist, id)| (id, (1.0 - dist).clamp(-1.0, 1.0)))
            .collect()
    }

    /// Batch parallel search using Rayon for high throughput query evaluation.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k, ef_search))
            .collect()
    }
}

/// Exact Brute-Force Linear Scan baseline engine for verification and recall measurement.
pub struct BruteForceScan;

impl BruteForceScan {
    /// Exact linear scan across unquantized FP32 vectors.
    pub fn search_fp32(database: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = database
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let sim: f32 = vec.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
                (idx, sim.clamp(-1.0, 1.0))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Parallel exact linear scan using Rayon.
    pub fn search_fp32_parallel(
        database: &[Vec<f32>],
        query: &[f32],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = database
            .par_iter()
            .enumerate()
            .map(|(idx, vec)| {
                let sim: f32 = vec.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
                (idx, sim.clamp(-1.0, 1.0))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Brute-force linear scan over SQ8 quantized vectors.
    pub fn search_sq8(database: &[Sq8Vector], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = database
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, vec.cosine_similarity_f32(query)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Evaluation utilities for benchmarking SQ8 and HNSW precision.
pub struct VectorEvalSuite;

impl VectorEvalSuite {
    /// Generate synthetic L2-normalized 384-dimensional unit vectors
    /// matching the distribution of MiniLM/Model2Vec embeddings on the unit sphere S^383.
    pub fn generate_synthetic_unit_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = FastRng::new(seed);
        let mut vectors = Vec::with_capacity(count);

        for _ in 0..count {
            let mut v = Vec::with_capacity(dim);
            let mut sum_sq = 0.0f32;

            // Generate Gaussian coordinates using Box-Muller
            for _ in 0..(dim / 2) {
                let (g1, g2) = rng.next_gaussian();
                v.push(g1);
                v.push(g2);
                sum_sq += g1 * g1 + g2 * g2;
            }
            if dim % 2 != 0 {
                let (g1, _) = rng.next_gaussian();
                v.push(g1);
                sum_sq += g1 * g1;
            }

            // Normalize to unit sphere
            let norm = sum_sq.sqrt().max(1e-12);
            let inv_norm = 1.0 / norm;
            for x in &mut v {
                *x *= inv_norm;
            }
            vectors.push(v);
        }

        vectors
    }

    /// Compute average and maximum cosine reconstruction error across `num_pairs` random vector pairs.
    ///
    /// For two vectors `u` and `v`, the reconstruction error is `|cos(u, v) - cos(u_sq8, v_sq8)|`.
    pub fn evaluate_sq8_reconstruction_error(
        vectors: &[Vec<f32>],
        num_pairs: usize,
        seed: u64,
    ) -> (f32, f32) {
        let n = vectors.len();
        assert!(n >= 2, "Need at least 2 vectors to evaluate pairs");

        let mut rng = FastRng::new(seed);
        let sq8_vectors: Vec<Sq8Vector> = vectors.iter().map(|v| Sq8Vector::encode(v)).collect();

        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;

        for _ in 0..num_pairs {
            let i = (rng.next_u64() as usize) % n;
            let mut j = (rng.next_u64() as usize) % n;
            if i == j {
                j = (j + 1) % n;
            }

            // Exact cosine
            let exact_cos: f32 = vectors[i]
                .iter()
                .zip(vectors[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            // SQ8 cosine
            let sq8_cos = sq8_vectors[i].cosine_similarity(&sq8_vectors[j]);

            let err = (exact_cos - sq8_cos).abs();
            total_error += err as f64;
            max_error = max_error.max(err);
        }

        let avg_error = (total_error / (num_pairs as f64)) as f32;
        (avg_error, max_error)
    }

    /// Compute Recall@K between ANN result IDs and ground-truth exact scan IDs.
    ///
    /// Recall@K = |ANN_top_K ∩ Exact_top_K| / K
    pub fn compute_recall(ann_results: &[(usize, f32)], exact_results: &[(usize, f32)], k: usize) -> f32 {
        let k_ann = ann_results.len().min(k);
        let k_exact = exact_results.len().min(k);
        if k_exact == 0 {
            return 1.0;
        }

        let mut matches = 0usize;
        for i in 0..k_ann {
            let ann_id = ann_results[i].0;
            if exact_results.iter().take(k_exact).any(|&(eid, _)| eid == ann_id) {
                matches += 1;
            }
        }

        matches as f32 / (k_exact as f32)
    }

    /// Benchmark report containing recall and latency metrics across multiple ef_search values.
    pub fn run_evaluation_benchmark(
        database_size: usize,
        dim: usize,
        num_queries: usize,
        ef_search_sweep: &[usize],
    ) -> BenchmarkReport {
        println!(
            "Generating {} synthetic {}-dimensional unit vectors...",
            database_size, dim
        );
        let dataset = Self::generate_synthetic_unit_vectors(database_size, dim, 42);
        let queries = Self::generate_synthetic_unit_vectors(num_queries, dim, 9999);

        // 1. SQ8 Reconstruction Error across 1,000 vector pairs
        println!("Evaluating SQ8 scalar quantization reconstruction error across 1,000 pairs...");
        let (avg_sq8_err, max_sq8_err) =
            Self::evaluate_sq8_reconstruction_error(&dataset, 1000, 12345);

        // 2. Build HNSW index
        println!("Building HNSW index with SQ8 quantization (ef_construction=64, M=16)...");
        let start_build = Instant::now();
        let mut index = HnswIndex::with_default_config();
        for (idx, vec) in dataset.iter().enumerate() {
            index.insert(idx, vec);
        }
        let build_time = start_build.elapsed();
        let memory_bytes = index.memory_usage_bytes();

        // 3. Ground-truth exact linear scans
        println!("Computing exact brute-force linear scans for {} queries...", num_queries);
        let mut ground_truth_50 = Vec::with_capacity(num_queries);
        let mut brute_force_latencies_us = Vec::with_capacity(num_queries);

        for q in &queries {
            let start_q = Instant::now();
            let exact = BruteForceScan::search_fp32(&dataset, q, 50);
            let elapsed_us = start_q.elapsed().as_micros() as f64;
            brute_force_latencies_us.push(elapsed_us);
            ground_truth_50.push(exact);
        }

        let bf_mean_us = brute_force_latencies_us.iter().sum::<f64>() / (num_queries as f64);
        let mut sorted_bf = brute_force_latencies_us.clone();
        sorted_bf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bf_p50 = sorted_bf[sorted_bf.len() * 50 / 100];
        let bf_p95 = sorted_bf[sorted_bf.len() * 95 / 100];
        let bf_p99 = sorted_bf[sorted_bf.len() * 99 / 100];

        // 4. Sweep ef_search and record recall + latency
        let mut sweep_results = Vec::new();

        for &ef in ef_search_sweep {
            let mut recall_10_sum = 0.0f32;
            let mut recall_50_sum = 0.0f32;
            let mut latencies_us = Vec::with_capacity(num_queries);

            for (i, q) in queries.iter().enumerate() {
                let start_q = Instant::now();
                let ann_results = index.search(q, 50, ef);
                let elapsed_us = start_q.elapsed().as_micros() as f64;
                latencies_us.push(elapsed_us);

                let r10 = Self::compute_recall(&ann_results, &ground_truth_50[i], 10);
                let r50 = Self::compute_recall(&ann_results, &ground_truth_50[i], 50);

                recall_10_sum += r10;
                recall_50_sum += r50;
            }

            let avg_r10 = recall_10_sum / (num_queries as f32);
            let avg_r50 = recall_50_sum / (num_queries as f32);

            let mean_us = latencies_us.iter().sum::<f64>() / (num_queries as f64);
            let mut sorted_lat = latencies_us.clone();
            sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_us = sorted_lat[sorted_lat.len() * 50 / 100];
            let p95_us = sorted_lat[sorted_lat.len() * 95 / 100];
            let p99_us = sorted_lat[sorted_lat.len() * 99 / 100];
            let qps = 1_000_000.0 / mean_us;
            let speedup = bf_mean_us / mean_us;

            sweep_results.push(SweepPoint {
                ef_search: ef,
                recall_at_10: avg_r10,
                recall_at_50: avg_r50,
                mean_latency_us: mean_us,
                p50_latency_us: p50_us,
                p95_latency_us: p95_us,
                p99_latency_us: p99_us,
                qps,
                speedup_factor: speedup,
            });
        }

        BenchmarkReport {
            num_vectors: database_size,
            dim,
            num_queries,
            sq8_avg_cosine_error: avg_sq8_err,
            sq8_max_cosine_error: max_sq8_err,
            build_time_ms: build_time.as_millis() as f64,
            memory_usage_mb: memory_bytes as f64 / (1024.0 * 1024.0),
            brute_force_mean_us: bf_mean_us,
            brute_force_p50_us: bf_p50,
            brute_force_p95_us: bf_p95,
            brute_force_p99_us: bf_p99,
            sweep_results,
        }
    }
}

/// Results point for an ef_search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepPoint {
    pub ef_search: usize,
    pub recall_at_10: f32,
    pub recall_at_50: f32,
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub qps: f64,
    pub speedup_factor: f64,
}

/// Comprehensive benchmarking report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub num_vectors: usize,
    pub dim: usize,
    pub num_queries: usize,
    pub sq8_avg_cosine_error: f32,
    pub sq8_max_cosine_error: f32,
    pub build_time_ms: f64,
    pub memory_usage_mb: f64,
    pub brute_force_mean_us: f64,
    pub brute_force_p50_us: f64,
    pub brute_force_p95_us: f64,
    pub brute_force_p99_us: f64,
    pub sweep_results: Vec<SweepPoint>,
}
