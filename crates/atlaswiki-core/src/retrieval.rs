//! Hybrid Retrieval & Multi-Signal Reranking for AtlasWiki.
//!
//! Combines SQLite FTS5 BM25, dense vector cosine similarity, Reciprocal Rank Fusion (RRF),
//! and PKB-specific multipliers (title, heading, tag, and PageRank boosts).

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::embedder::EmbeddingEngine;
use crate::storage::StorageEngine;

/// Query classification intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    CodeSymbol,
    NaturalLanguage,
    BalancedHybrid,
}

/// Weights assigned based on detected query intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridWeights {
    pub bm25_weight: f32,
    pub vector_weight: f32,
    pub intent: QueryIntent,
}

/// Zero-latency heuristic query classifier.
pub struct QueryClassifier;

impl QueryClassifier {
    pub fn classify(query: &str) -> HybridWeights {
        let q = query.trim();

        // 1. Code symbol check
        let is_code = q.contains("::")
            || q.contains("->")
            || q.contains("()")
            || q.starts_with('#')
            || (q.starts_with('"') && q.ends_with('"'))
            || q.ends_with(".rs")
            || q.ends_with(".md")
            || q.ends_with(".ts")
            || q.ends_with(".py")
            || q.contains('_')
            || q.starts_with("fn ")
            || q.starts_with("struct ")
            || q.starts_with("class ")
            || q.starts_with("impl ")
            || q.starts_with("enum ")
            || q.starts_with("trait ")
            || q.starts_with("interface ")
            || q.starts_with("type ")
            || q.starts_with("def ")
            || q.starts_with("let ");

        if is_code {
            return HybridWeights {
                bm25_weight: 0.85,
                vector_weight: 0.15,
                intent: QueryIntent::CodeSymbol,
            };
        }

        // 2. Natural language / question check
        let lower = q.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let is_nl = lower.ends_with('?')
            || words.iter().any(|&w| {
                matches!(
                    w,
                    "how" | "why" | "what" | "where" | "when" | "explain" | "describe" | "difference"
                )
            })
            || (words.len() >= 5);

        if is_nl {
            return HybridWeights {
                bm25_weight: 0.20,
                vector_weight: 0.80,
                intent: QueryIntent::NaturalLanguage,
            };
        }

        // 3. Balanced hybrid default
        HybridWeights {
            bm25_weight: 0.50,
            vector_weight: 0.50,
            intent: QueryIntent::BalancedHybrid,
        }
    }
}

/// Configuration for Reciprocal Rank Fusion.
#[derive(Debug, Clone)]
pub struct RrfConfig {
    pub k: u32, // Default 60 for PKBs
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60 }
    }
}

/// Final search result returned by the retriever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub doc_id: String,
    pub title: String,
    pub breadcrumbs: String,
    pub snippet: String,
    pub score: f32,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub pagerank: f32,
}

/// Multi-signal domain reranker with PKB multipliers.
pub struct PkbReranker {
    pub title_exact_multiplier: f32,
    pub title_partial_multiplier: f32,
    pub heading_exact_multiplier: f32,
    pub heading_partial_multiplier: f32,
    pub tag_match_multiplier: f32,
    pub max_pagerank_boost: f32,
    pub max_composite_multiplier: f32,
}

impl Default for PkbReranker {
    fn default() -> Self {
        Self {
            title_exact_multiplier: 2.5,
            title_partial_multiplier: 1.6,
            heading_exact_multiplier: 2.0,
            heading_partial_multiplier: 1.3,
            tag_match_multiplier: 1.5,
            max_pagerank_boost: 0.20,
            max_composite_multiplier: 5.0,
        }
    }
}

impl PkbReranker {
    pub fn compute_boost(
        &self,
        query: &str,
        title: &str,
        breadcrumbs: &str,
        tags: &[String],
        pagerank: f32,
    ) -> f32 {
        let q_clean = query.trim().to_lowercase();
        let t_clean = title.trim().to_lowercase();
        let b_clean = breadcrumbs.trim().to_lowercase();

        // 1. Title match
        let title_boost = if t_clean == q_clean {
            self.title_exact_multiplier
        } else if t_clean.contains(&q_clean) || q_clean.contains(&t_clean) {
            self.title_partial_multiplier
        } else {
            1.0
        };

        // 2. Heading match
        let heading_boost = if b_clean.ends_with(&q_clean) {
            self.heading_exact_multiplier
        } else if b_clean.contains(&q_clean) {
            self.heading_partial_multiplier
        } else {
            1.0
        };

        // 3. Tag match
        let tag_match = tags.iter().any(|t| {
            let t_clean = t.trim_start_matches('#').to_lowercase();
            t_clean == q_clean || q_clean.contains(&t_clean)
        });
        let tag_boost = if tag_match {
            self.tag_match_multiplier
        } else {
            1.0
        };

        // 4. PageRank boost (+0% to +20%)
        let pr_boost = 1.0 + self.max_pagerank_boost * pagerank.clamp(0.0, 1.0);

        // Composite capped multiplier
        let composite = title_boost * heading_boost * tag_boost * pr_boost;
        composite.min(self.max_composite_multiplier)
    }
}

/// Unified Hybrid Retriever.
pub struct HybridRetriever {
    storage: StorageEngine,
    embedder: EmbeddingEngine,
    rrf_config: RrfConfig,
    reranker: PkbReranker,
}

impl HybridRetriever {
    pub fn new(storage: StorageEngine, embedder: EmbeddingEngine) -> Self {
        Self {
            storage,
            embedder,
            rrf_config: RrfConfig::default(),
            reranker: PkbReranker::default(),
        }
    }

    /// Search the knowledge base using hybrid multi-signal retrieval.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let weights = QueryClassifier::classify(query);
        let candidate_pool = (limit * 3).max(30);

        // 1. Lexical retrieval via SQLite FTS5 BM25
        let fts_query = sanitize_fts5_query(query);
        let fts_results = if !fts_query.is_empty() {
            self.storage.search_fts(&fts_query, candidate_pool).unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut bm25_ranks: HashMap<String, (usize, String, String, String)> = HashMap::new();
        for (rank, (chunk_id, title, breadcrumbs, snippet, _score)) in fts_results.into_iter().enumerate() {
            bm25_ranks.insert(chunk_id, (rank + 1, title, breadcrumbs, snippet));
        }

        // 2. Vector semantic retrieval via Model2Vec (if enabled)
        let mut vector_ranks: HashMap<String, usize> = HashMap::new();
        if let Some(q_vec) = self.embedder.embed_text(query) {
            let vec_results = self.storage.search_vectors(&q_vec, candidate_pool).unwrap_or_default();
            for (rank, (chunk_id, _sim)) in vec_results.into_iter().enumerate() {
                vector_ranks.insert(chunk_id, rank + 1);
            }
        }

        // 3. Reciprocal Rank Fusion
        let mut candidate_ids: Vec<String> = bm25_ranks.keys().cloned().collect();
        for id in vector_ranks.keys() {
            if !bm25_ranks.contains_key(id) {
                candidate_ids.push(id.clone());
            }
        }

        let k = self.rrf_config.k as f32;
        let mut scored_results = Vec::new();

        for chunk_id in candidate_ids {
            let r_bm25 = bm25_ranks.get(&chunk_id).map(|(r, _, _, _)| *r);
            let r_vec = vector_ranks.get(&chunk_id).copied();

            let bm25_rrf = r_bm25.map(|r| 1.0 / (k + r as f32)).unwrap_or(0.0);
            let vec_rrf = r_vec.map(|r| 1.0 / (k + r as f32)).unwrap_or(0.0);

            let fused_score = weights.bm25_weight * bm25_rrf + weights.vector_weight * vec_rrf;

            let (title, breadcrumbs, snippet) = if let Some((_, t, b, s)) = bm25_ranks.get(&chunk_id) {
                (t.clone(), b.clone(), s.clone())
            } else {
                // Fetch from storage chunks table
                self.storage.with_read_conn(|conn| {
                    let mut stmt = conn.prepare_cached(
                        "SELECT title, breadcrumbs, content FROM chunks WHERE chunk_id = ?1",
                    )?;
                    let mut rows = stmt.query([&chunk_id])?;
                    if let Some(row) = rows.next()? {
                        let t: String = row.get(0)?;
                        let b: String = row.get(1)?;
                        let c: String = row.get(2)?;
                        let snip = c.chars().take(200).collect::<String>();
                        Ok((t, b, snip))
                    } else {
                        Ok((chunk_id.clone(), String::new(), String::new()))
                    }
                }).unwrap_or((chunk_id.clone(), String::new(), String::new()))
            };

            // Calculate domain boost
            let boost = self.reranker.compute_boost(query, &title, &breadcrumbs, &[], 0.0);
            let final_score = fused_score * boost;

            scored_results.push(SearchResult {
                chunk_id,
                doc_id: title.clone(),
                title,
                breadcrumbs,
                snippet,
                score: final_score,
                bm25_rank: r_bm25,
                vector_rank: r_vec,
                pagerank: 0.0,
            });
        }

        // Sort descending by score
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored_results.truncate(limit);

        Ok(scored_results)
    }
}

/// Cleans and formats query string for SQLite FTS5 MATCH syntax.
pub fn sanitize_fts5_query(raw: &str) -> String {
    let tokens: Vec<&str> = raw
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-'))
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return String::new();
    }

    tokens
        .iter()
        .map(|t| format!("\"{t}\"*"))
        .collect::<Vec<_>>()
        .join(" OR ")
}
