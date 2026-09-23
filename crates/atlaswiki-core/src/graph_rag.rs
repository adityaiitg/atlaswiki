//! Graph RAG & Multi-Hop Context Extraction Engine for AtlasWiki.
//!
//! Provides Steiner-tree connective subgraph extraction, K-hop connective path discovery,
//! and linearized markdown context serialization for LLM retrieval-augmented generation.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use atlaswiki_parser::ast::{LinkType, ParsedDocument};
use crate::graph::NodeType;
use crate::storage::StorageEngine;

/// Directed edge within an extracted reasoning path or Steiner tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f32,
}

/// A multi-hop connective path between notes/concepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<String>,
    pub edges: Vec<PathEdge>,
    pub total_weight: f32,
    pub hop_count: usize,
}

impl GraphPath {
    /// Formats the path into a linearized representation:
    /// `[[Node A]] --[wikilink]--> [[Node B]] --[tag:#security]--> [[Node C]]`
    pub fn to_linearized_string(&self) -> String {
        if self.nodes.is_empty() {
            return String::new();
        }
        if self.edges.is_empty() {
            return format!("[[{}]]", self.nodes[0]);
        }

        let mut out = format!("[[{}]]", self.nodes[0]);
        for (i, edge) in self.edges.iter().enumerate() {
            let next_node = self
                .nodes
                .get(i + 1)
                .map(|s| s.as_str())
                .unwrap_or(&edge.target);
            out.push_str(&format!(" --[{}]--> [[{}]]", edge.relation, next_node));
        }
        out
    }
}

/// Minimal Steiner tree spanning a set of terminal concept seeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteinerTree {
    pub terminal_nodes: Vec<String>,
    pub steiner_nodes: Vec<String>,
    pub edges: Vec<PathEdge>,
    pub total_weight: f32,
}

/// Note context metadata embedded into Graph RAG prompts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeContext {
    pub title: String,
    pub path: Option<String>,
    pub tags: Vec<String>,
    pub word_count: usize,
    pub pagerank: f32,
    pub snippet: Option<String>,
}

/// Full Graph RAG extraction response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagResult {
    pub seeds: Vec<String>,
    pub paths: Vec<GraphPath>,
    pub steiner_tree: Option<SteinerTree>,
    pub involved_nodes: Vec<NodeContext>,
    pub markdown_context: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct InternalNode {
    title: String,
    path: Option<PathBuf>,
    node_type: NodeType,
    tags: Vec<String>,
    word_count: usize,
    pagerank: f32,
    snippet: Option<String>,
}

#[derive(Debug, Clone)]
struct InternalEdge {
    relation: String,
    weight: f32,
}

/// Priority queue state for Dijkstra search.
#[derive(Clone, PartialEq)]
struct DijkstraState {
    cost: f32,
    node: NodeIndex,
}

impl Eq for DijkstraState {}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Graph RAG multi-hop connective reasoning engine.
pub struct GraphRagEngine {
    graph: DiGraph<InternalNode, InternalEdge>,
    title_map: HashMap<String, NodeIndex>,
    alias_map: HashMap<String, NodeIndex>,
}

impl Default for GraphRagEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRagEngine {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            title_map: HashMap::new(),
            alias_map: HashMap::new(),
        }
    }

    /// Construct engine from an array of parsed documents.
    pub fn from_documents(docs: &[ParsedDocument]) -> Self {
        let mut engine = Self::new();
        let mut tag_to_nodes: HashMap<String, Vec<NodeIndex>> = HashMap::new();

        // Pass 1: Insert Document nodes
        for doc in docs {
            let tags: Vec<String> = doc.tags.iter().map(|t| t.name.clone()).collect();
            let snippet = doc
                .chunks
                .first()
                .map(|c| c.content.clone())
                .or_else(|| doc.sections.first().map(|s| s.content.clone()));

            let node = InternalNode {
                title: doc.title.clone(),
                path: Some(doc.path.clone()),
                node_type: NodeType::Document,
                tags: tags.clone(),
                word_count: doc.word_count,
                pagerank: 0.0,
                snippet,
            };

            let idx = engine.graph.add_node(node);
            engine.title_map.insert(doc.title.to_lowercase(), idx);

            for alias in &doc.frontmatter.aliases {
                engine.alias_map.insert(alias.to_lowercase(), idx);
            }

            for tag in tags {
                let clean_tag = tag.trim_start_matches('#').to_lowercase();
                tag_to_nodes.entry(clean_tag).or_default().push(idx);
            }
        }

        // Pass 2: Insert direct link edges
        for doc in docs {
            let source_idx = match engine.title_map.get(&doc.title.to_lowercase()) {
                Some(&idx) => idx,
                None => continue,
            };

            for link in &doc.links {
                let target_title = link.target_note.trim();
                if target_title.is_empty() {
                    continue;
                }

                let target_idx = engine.resolve_or_create_dangling(target_title);

                let (rel_type, weight) = match link.link_type {
                    LinkType::Wikilink => ("wikilink".to_string(), 1.0f32),
                    LinkType::Embed => ("embed".to_string(), 0.9f32),
                    LinkType::Markdown => ("markdown".to_string(), 1.2f32),
                };

                engine.graph.add_edge(
                    source_idx,
                    target_idx,
                    InternalEdge {
                        relation: rel_type,
                        weight,
                    },
                );

                // Add reverse backlink edge with slightly higher weight for bidirectional traversal
                engine.graph.add_edge(
                    target_idx,
                    source_idx,
                    InternalEdge {
                        relation: "backlink".to_string(),
                        weight: weight + 0.3,
                    },
                );
            }
        }

        // Pass 3: Insert tag co-occurrence edges (shared tags)
        // Format: `tag:#<tag_name>`
        for (tag_name, node_indices) in tag_to_nodes {
            if node_indices.len() > 1 && node_indices.len() <= 50 {
                let rel = format!("tag:#{}", tag_name);
                for i in 0..node_indices.len() {
                    for j in (i + 1)..node_indices.len() {
                        let u = node_indices[i];
                        let v = node_indices[j];
                        if u != v {
                            engine.graph.add_edge(
                                u,
                                v,
                                InternalEdge {
                                    relation: rel.clone(),
                                    weight: 1.5,
                                },
                            );
                            engine.graph.add_edge(
                                v,
                                u,
                                InternalEdge {
                                    relation: rel.clone(),
                                    weight: 1.5,
                                },
                            );
                        }
                    }
                }
            }
        }

        // Pass 4: Compute normalized PageRank
        engine.compute_pagerank(0.85, 30);

        engine
    }

    /// Construct engine directly from SQLite storage.
    pub fn from_storage(storage: &StorageEngine) -> Result<Self> {
        let mut engine = Self::new();
        let mut tag_to_nodes: HashMap<String, Vec<NodeIndex>> = HashMap::new();

        storage.with_read_conn(|conn| {
            // Read documents
            let mut stmt_docs = conn.prepare(
                "SELECT doc_id, path, title, word_count, frontmatter_json FROM documents",
            )?;
            let doc_rows = stmt_docs.query_map([], |row| {
                let doc_id: String = row.get(0)?;
                let path_str: String = row.get(1)?;
                let title: String = row.get(2)?;
                let word_count: i64 = row.get(3)?;
                let fm_json: Option<String> = row.get(4)?;
                Ok((doc_id, path_str, title, word_count as usize, fm_json))
            })?;

            let mut doc_id_to_idx: HashMap<String, NodeIndex> = HashMap::new();

            for row in doc_rows {
                let (doc_id, path_str, title, word_count, fm_json) = row?;
                let mut aliases = Vec::new();
                if let Some(raw_fm) = fm_json {
                    if let Ok(fm) = serde_json::from_str::<atlaswiki_parser::ast::Frontmatter>(&raw_fm) {
                        aliases = fm.aliases;
                    }
                }

                let node = InternalNode {
                    title: title.clone(),
                    path: Some(PathBuf::from(&path_str)),
                    node_type: NodeType::Document,
                    tags: Vec::new(),
                    word_count,
                    pagerank: 0.0,
                    snippet: None,
                };

                let idx = engine.graph.add_node(node);
                engine.title_map.insert(title.to_lowercase(), idx);
                doc_id_to_idx.insert(doc_id, idx);

                for alias in aliases {
                    engine.alias_map.insert(alias.to_lowercase(), idx);
                }
            }

            // Read tags
            let mut stmt_tags = conn.prepare("SELECT doc_id, tag FROM tags")?;
            let tag_rows = stmt_tags.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            for r in tag_rows {
                let (doc_id, tag) = r?;
                if let Some(&idx) = doc_id_to_idx.get(&doc_id) {
                    if let Some(node) = engine.graph.node_weight_mut(idx) {
                        node.tags.push(tag.clone());
                    }
                    let clean = tag.trim_start_matches('#').to_lowercase();
                    tag_to_nodes.entry(clean).or_default().push(idx);
                }
            }

            // Read links
            let mut stmt_links = conn.prepare(
                "SELECT source_doc_id, target_note, link_type, context_snippet FROM links",
            )?;
            let link_rows = stmt_links.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?;

            for r in link_rows {
                let (source_doc_id, target_note, link_type, _snippet) = r?;
                let source_idx = match doc_id_to_idx.get(&source_doc_id) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let target_idx = engine.resolve_or_create_dangling(&target_note);

                let (rel_type, weight) = match link_type.as_str() {
                    "embed" => ("embed".to_string(), 0.9f32),
                    "markdown" => ("markdown".to_string(), 1.2f32),
                    _ => ("wikilink".to_string(), 1.0f32),
                };

                engine.graph.add_edge(
                    source_idx,
                    target_idx,
                    InternalEdge {
                        relation: rel_type,
                        weight,
                    },
                );

                engine.graph.add_edge(
                    target_idx,
                    source_idx,
                    InternalEdge {
                        relation: "backlink".to_string(),
                        weight: weight + 0.3,
                    },
                );
            }

            // Read first chunk as snippet
            let mut stmt_chunks = conn.prepare(
                "SELECT doc_id, content FROM chunks GROUP BY doc_id",
            )?;
            let chunk_rows = stmt_chunks.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            for r in chunk_rows {
                let (doc_id, content) = r?;
                if let Some(&idx) = doc_id_to_idx.get(&doc_id) {
                    if let Some(node) = engine.graph.node_weight_mut(idx) {
                        node.snippet = Some(content);
                    }
                }
            }

            Ok(())
        })?;

        // Shared tags
        for (tag_name, node_indices) in tag_to_nodes {
            if node_indices.len() > 1 && node_indices.len() <= 50 {
                let rel = format!("tag:#{}", tag_name);
                for i in 0..node_indices.len() {
                    for j in (i + 1)..node_indices.len() {
                        let u = node_indices[i];
                        let v = node_indices[j];
                        if u != v {
                            engine.graph.add_edge(
                                u,
                                v,
                                InternalEdge {
                                    relation: rel.clone(),
                                    weight: 1.5,
                                },
                            );
                            engine.graph.add_edge(
                                v,
                                u,
                                InternalEdge {
                                    relation: rel.clone(),
                                    weight: 1.5,
                                },
                            );
                        }
                    }
                }
            }
        }

        engine.compute_pagerank(0.85, 30);
        Ok(engine)
    }

    /// Resolves target note to NodeIndex, creating dangling node if absent.
    fn resolve_or_create_dangling(&mut self, target_note: &str) -> NodeIndex {
        let lower = target_note.to_lowercase();
        if let Some(&idx) = self.title_map.get(&lower) {
            return idx;
        }
        if let Some(&idx) = self.alias_map.get(&lower) {
            return idx;
        }

        let node = InternalNode {
            title: target_note.to_string(),
            path: None,
            node_type: NodeType::Dangling,
            tags: Vec::new(),
            word_count: 0,
            pagerank: 0.0,
            snippet: None,
        };

        let idx = self.graph.add_node(node);
        self.title_map.insert(lower, idx);
        idx
    }

    /// Resolve an input seed string to a graph NodeIndex.
    pub fn resolve_seed(&self, seed: &str) -> Option<NodeIndex> {
        let clean = seed.trim();
        let lower = clean.to_lowercase();

        // 1. Direct title match
        if let Some(&idx) = self.title_map.get(&lower) {
            return Some(idx);
        }
        // 2. Direct alias match
        if let Some(&idx) = self.alias_map.get(&lower) {
            return Some(idx);
        }
        // 3. Strip wikilink brackets [[Note]]
        if clean.starts_with("[[") && clean.ends_with("]]") {
            let inner = &clean[2..clean.len() - 2];
            let inner_lower = inner.to_lowercase();
            if let Some(&idx) = self.title_map.get(&inner_lower) {
                return Some(idx);
            }
            if let Some(&idx) = self.alias_map.get(&inner_lower) {
                return Some(idx);
            }
        }
        // 4. Tag match: e.g. #tag or tag
        let tag_name = clean.trim_start_matches('#').to_lowercase();
        for idx in self.graph.node_indices() {
            if self.graph[idx]
                .tags
                .iter()
                .any(|t| t.trim_start_matches('#').eq_ignore_ascii_case(&tag_name))
            {
                return Some(idx);
            }
        }

        None
    }

    /// Compute PageRank centrality.
    pub fn compute_pagerank(&mut self, damping: f32, max_iterations: usize) {
        let n = self.graph.node_count();
        if n == 0 {
            return;
        }

        let mut pr = vec![1.0 / n as f32; n];
        let mut out_degrees = vec![0usize; n];

        for i in 0..n {
            let idx = NodeIndex::new(i);
            out_degrees[i] = self.graph.neighbors_directed(idx, Direction::Outgoing).count();
        }

        for _ in 0..max_iterations {
            let mut next_pr = vec![0.0f32; n];
            let mut dangling_sum = 0.0f32;

            for i in 0..n {
                if out_degrees[i] == 0 {
                    dangling_sum += pr[i];
                }
            }

            let base = (1.0 - damping + damping * dangling_sum) / n as f32;

            for i in 0..n {
                let target_idx = NodeIndex::new(i);
                let mut incoming_sum = 0.0f32;

                for edge in self.graph.edges_directed(target_idx, Direction::Incoming) {
                    let source = edge.source().index();
                    let out_deg = out_degrees[source];
                    if out_deg > 0 {
                        incoming_sum += pr[source] / out_deg as f32;
                    }
                }

                next_pr[i] = base + damping * incoming_sum;
            }

            pr = next_pr;
        }

        let max_pr = pr.iter().cloned().fold(0.0f32, f32::max);
        let min_pr = pr.iter().cloned().fold(1.0f32, f32::min);
        let range = (max_pr - min_pr).max(1e-6);

        for i in 0..n {
            let norm_score = ((pr[i] - min_pr) / range).clamp(0.0, 1.0);
            let idx = NodeIndex::new(i);
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.pagerank = norm_score;
            }
        }
    }

    /// Extract K-hop connective paths between concept seeds.
    pub fn extract_paths(
        &self,
        seeds: &[&str],
        max_hops: usize,
        max_paths: usize,
    ) -> Vec<GraphPath> {
        let mut seed_indices = Vec::new();
        for &s in seeds {
            if let Some(idx) = self.resolve_seed(s) {
                if !seed_indices.contains(&idx) {
                    seed_indices.push(idx);
                }
            }
        }

        if seed_indices.is_empty() {
            return Vec::new();
        }

        let mut all_paths: Vec<GraphPath> = Vec::new();

        if seed_indices.len() == 1 {
            // Single seed: Extract 1-hop and 2-hop radial neighborhood paths
            let root = seed_indices[0];
            self.extract_radial_paths(root, max_hops.min(2), max_paths, &mut all_paths);
        } else {
            // Pairwise path search between all distinct seed pairs
            for i in 0..seed_indices.len() {
                for j in (i + 1)..seed_indices.len() {
                    let src = seed_indices[i];
                    let dst = seed_indices[j];

                    let pair_paths = self.find_simple_paths(src, dst, max_hops, max_paths);
                    all_paths.extend(pair_paths);
                }
            }
        }

        // Deduplicate paths by linearized representation
        let mut seen = HashSet::new();
        let mut unique_paths = Vec::new();

        all_paths.sort_by(|a, b| {
            a.total_weight
                .partial_cmp(&b.total_weight)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.hop_count.cmp(&b.hop_count))
        });

        for p in all_paths {
            let key = p.to_linearized_string();
            if seen.insert(key) {
                unique_paths.push(p);
                if unique_paths.len() >= max_paths {
                    break;
                }
            }
        }

        unique_paths
    }

    /// Find simple paths between src and dst up to max_hops using bounded DFS/BFS.
    fn find_simple_paths(
        &self,
        src: NodeIndex,
        dst: NodeIndex,
        max_hops: usize,
        limit: usize,
    ) -> Vec<GraphPath> {
        let mut results = Vec::new();
        if src == dst {
            results.push(GraphPath {
                nodes: vec![self.graph[src].title.clone()],
                edges: Vec::new(),
                total_weight: 0.0,
                hop_count: 0,
            });
            return results;
        }

        // Queue item: (curr_node, visited_nodes, accumulated_edges, accumulated_weight)
        let mut queue: VecDeque<(NodeIndex, Vec<NodeIndex>, Vec<PathEdge>, f32)> = VecDeque::new();
        queue.push_back((src, vec![src], Vec::new(), 0.0));

        while let Some((curr, visited, edges, weight)) = queue.pop_front() {
            if curr == dst {
                let node_names = visited
                    .iter()
                    .map(|&idx| self.graph[idx].title.clone())
                    .collect();

                results.push(GraphPath {
                    nodes: node_names,
                    edges,
                    total_weight: weight,
                    hop_count: visited.len() - 1,
                });

                if results.len() >= limit {
                    break;
                }
                continue;
            }

            if visited.len() - 1 >= max_hops {
                continue;
            }

            for edge in self.graph.edges_directed(curr, Direction::Outgoing) {
                let next = edge.target();
                if visited.contains(&next) {
                    continue;
                }

                let edge_data = edge.weight();
                let mut next_visited = visited.clone();
                next_visited.push(next);

                let mut next_edges = edges.clone();
                next_edges.push(PathEdge {
                    source: self.graph[curr].title.clone(),
                    target: self.graph[next].title.clone(),
                    relation: edge_data.relation.clone(),
                    weight: edge_data.weight,
                });

                queue.push_back((
                    next,
                    next_visited,
                    next_edges,
                    weight + edge_data.weight,
                ));
            }
        }

        results
    }

    /// Extract radial neighbor chains for single-seed queries.
    fn extract_radial_paths(
        &self,
        root: NodeIndex,
        max_hops: usize,
        limit: usize,
        out: &mut Vec<GraphPath>,
    ) {
        let mut queue: VecDeque<(NodeIndex, Vec<NodeIndex>, Vec<PathEdge>, f32)> = VecDeque::new();
        queue.push_back((root, vec![root], Vec::new(), 0.0));

        while let Some((curr, visited, edges, weight)) = queue.pop_front() {
            if visited.len() > 1 {
                let node_names = visited
                    .iter()
                    .map(|&idx| self.graph[idx].title.clone())
                    .collect();

                out.push(GraphPath {
                    nodes: node_names,
                    edges: edges.clone(),
                    total_weight: weight,
                    hop_count: visited.len() - 1,
                });

                if out.len() >= limit {
                    break;
                }
            }

            if visited.len() - 1 >= max_hops {
                continue;
            }

            for edge in self.graph.edges_directed(curr, Direction::Outgoing) {
                let next = edge.target();
                if visited.contains(&next) {
                    continue;
                }

                // Avoid traversing backlink in radial search to keep clean forward paths
                if edge.weight().relation == "backlink" {
                    continue;
                }

                let mut next_visited = visited.clone();
                next_visited.push(next);

                let mut next_edges = edges.clone();
                next_edges.push(PathEdge {
                    source: self.graph[curr].title.clone(),
                    target: self.graph[next].title.clone(),
                    relation: edge.weight().relation.clone(),
                    weight: edge.weight().weight,
                });

                queue.push_back((
                    next,
                    next_visited,
                    next_edges,
                    weight + edge.weight().weight,
                ));
            }
        }
    }

    /// Extract minimal Steiner tree connecting the terminal seeds.
    ///
    /// Implements the Takahashi-Matsuyama heuristic: greedily connects
    /// the closest unconnected terminal to the current tree via shortest path.
    pub fn extract_steiner_tree(&self, seeds: &[&str]) -> Option<SteinerTree> {
        let mut terminal_indices = Vec::new();
        for &s in seeds {
            if let Some(idx) = self.resolve_seed(s) {
                if !terminal_indices.contains(&idx) {
                    terminal_indices.push(idx);
                }
            }
        }

        if terminal_indices.is_empty() {
            return None;
        }

        if terminal_indices.len() == 1 {
            let node_title = self.graph[terminal_indices[0]].title.clone();
            return Some(SteinerTree {
                terminal_nodes: vec![node_title],
                steiner_nodes: Vec::new(),
                edges: Vec::new(),
                total_weight: 0.0,
            });
        }

        let mut tree_nodes: HashSet<NodeIndex> = HashSet::new();
        let mut tree_edges: Vec<PathEdge> = Vec::new();
        let mut total_weight = 0.0f32;

        let mut terminals_left: HashSet<NodeIndex> = terminal_indices[1..].iter().copied().collect();
        tree_nodes.insert(terminal_indices[0]);

        while !terminals_left.is_empty() {
            // Run multi-source Dijkstra from all current tree_nodes
            let mut dist: HashMap<NodeIndex, f32> = HashMap::new();
            let mut prev: HashMap<NodeIndex, (NodeIndex, PathEdge)> = HashMap::new();
            let mut pq = BinaryHeap::new();

            for &root in &tree_nodes {
                dist.insert(root, 0.0);
                pq.push(DijkstraState { cost: 0.0, node: root });
            }

            let mut closest_terminal: Option<NodeIndex> = None;

            while let Some(DijkstraState { cost, node }) = pq.pop() {
                if cost > *dist.get(&node).unwrap_or(&f32::INFINITY) {
                    continue;
                }

                if terminals_left.contains(&node) {
                    closest_terminal = Some(node);
                    break;
                }

                for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                    let next = edge.target();
                    let edge_data = edge.weight();
                    let next_cost = cost + edge_data.weight;

                    if next_cost < *dist.get(&next).unwrap_or(&f32::INFINITY) {
                        dist.insert(next, next_cost);
                        prev.insert(
                            next,
                            (
                                node,
                                PathEdge {
                                    source: self.graph[node].title.clone(),
                                    target: self.graph[next].title.clone(),
                                    relation: edge_data.relation.clone(),
                                    weight: edge_data.weight,
                                },
                            ),
                        );
                        pq.push(DijkstraState {
                            cost: next_cost,
                            node: next,
                        });
                    }
                }
            }

            let target = match closest_terminal {
                Some(t) => t,
                None => break, // Remaining terminals are unreachable / disconnected
            };

            // Trace path back to tree_nodes
            let mut curr = target;
            let mut path_segment_edges = Vec::new();
            let mut path_segment_nodes = Vec::new();

            while !tree_nodes.contains(&curr) {
                path_segment_nodes.push(curr);
                if let Some((p_node, p_edge)) = prev.get(&curr) {
                    path_segment_edges.push(p_edge.clone());
                    curr = *p_node;
                } else {
                    break;
                }
            }

            path_segment_edges.reverse();
            for edge in path_segment_edges {
                total_weight += edge.weight;
                tree_edges.push(edge);
            }

            for node in path_segment_nodes {
                tree_nodes.insert(node);
            }

            terminals_left.remove(&target);
        }

        let terminal_set: HashSet<NodeIndex> = terminal_indices.iter().copied().collect();
        let steiner_nodes: Vec<String> = tree_nodes
            .iter()
            .filter(|idx| !terminal_set.contains(idx))
            .map(|idx| self.graph[*idx].title.clone())
            .collect();

        let terminal_nodes: Vec<String> = tree_nodes
            .iter()
            .filter(|idx| terminal_set.contains(idx))
            .map(|idx| self.graph[*idx].title.clone())
            .collect();

        Some(SteinerTree {
            terminal_nodes,
            steiner_nodes,
            edges: tree_edges,
            total_weight,
        })
    }

    /// Full Graph RAG extraction: extracts connective paths, Steiner tree,
    /// involved note metadata, and builds LLM-ready linearized markdown context.
    pub fn extract_context(
        &self,
        seeds: &[&str],
        max_hops: usize,
        max_paths: usize,
        include_content: bool,
    ) -> GraphRagResult {
        let paths = self.extract_paths(seeds, max_hops, max_paths);
        let steiner = self.extract_steiner_tree(seeds);

        // Collect all unique node titles involved in paths and Steiner tree
        let mut involved_titles = HashSet::new();
        for &s in seeds {
            involved_titles.insert(s.to_string());
        }
        for p in &paths {
            for n in &p.nodes {
                involved_titles.insert(n.clone());
            }
        }
        if let Some(ref st) = steiner {
            for t in &st.terminal_nodes {
                involved_titles.insert(t.clone());
            }
            for s in &st.steiner_nodes {
                involved_titles.insert(s.clone());
            }
        }

        let mut node_contexts = Vec::new();
        for title in involved_titles {
            if let Some(idx) = self.resolve_seed(&title) {
                let n = &self.graph[idx];
                let snippet = if include_content {
                    n.snippet.clone()
                } else {
                    None
                };

                node_contexts.push(NodeContext {
                    title: n.title.clone(),
                    path: n.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
                    tags: n.tags.clone(),
                    word_count: n.word_count,
                    pagerank: n.pagerank,
                    snippet,
                });
            }
        }

        node_contexts.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(Ordering::Equal)
        });

        let seed_strings: Vec<String> = seeds.iter().map(|s| s.to_string()).collect();
        let markdown_context = Self::build_markdown_context(
            &seed_strings,
            &paths,
            steiner.as_ref(),
            &node_contexts,
            include_content,
        );

        GraphRagResult {
            seeds: seed_strings,
            paths,
            steiner_tree: steiner,
            involved_nodes: node_contexts,
            markdown_context,
        }
    }

    /// Formats connective paths and node contexts into LLM-ready markdown blocks.
    pub fn build_markdown_context(
        seeds: &[String],
        paths: &[GraphPath],
        steiner: Option<&SteinerTree>,
        nodes: &[NodeContext],
        include_content: bool,
    ) -> String {
        let mut md = String::new();
        md.push_str("# Knowledge Graph Context\n\n");

        // Section 1: Connective Reasoning Paths
        md.push_str("## Connective Reasoning Paths\n");
        if paths.is_empty() {
            md.push_str(
                "- *No multi-hop connective paths found within specified hop boundary.*\n\n",
            );
        } else {
            for (i, p) in paths.iter().enumerate() {
                md.push_str(&format!("{}. `{}`\n", i + 1, p.to_linearized_string()));
            }
            md.push('\n');
        }

        // Section 2: Graph Context Summary
        md.push_str("## Graph Context Summary\n");
        let formatted_seeds = seeds
            .iter()
            .map(|s| format!("[[{}]]", s))
            .collect::<Vec<_>>()
            .join(", ");
        md.push_str(&format!("- **Seed Concepts**: {}\n", formatted_seeds));

        if let Some(st) = steiner {
            if !st.steiner_nodes.is_empty() {
                let formatted_steiner = st
                    .steiner_nodes
                    .iter()
                    .map(|s| format!("[[{}]]", s))
                    .collect::<Vec<_>>()
                    .join(", ");
                md.push_str(&format!(
                    "- **Steiner Bridge Concepts**: {}\n",
                    formatted_steiner
                ));
            }
            md.push_str(&format!(
                "- **Steiner Tree Weight**: {:.2}\n",
                st.total_weight
            ));
        }

        md.push_str(&format!("- **Total Connective Paths**: {}\n", paths.len()));
        md.push_str(&format!("- **Sub-graph Node Count**: {}\n\n", nodes.len()));

        // Section 3: Entity Details & Evidence
        if !nodes.is_empty() {
            md.push_str("## Entity Details & Evidence\n");
            for n in nodes {
                md.push_str(&format!("### [[{}]]\n", n.title));
                if let Some(ref p) = n.path {
                    md.push_str(&format!("- **File**: `{}`\n", p));
                }
                if !n.tags.is_empty() {
                    let tag_str = n
                        .tags
                        .iter()
                        .map(|t| {
                            if t.starts_with('#') {
                                t.clone()
                            } else {
                                format!("#{}", t)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    md.push_str(&format!("- **Tags**: {}\n", tag_str));
                }
                md.push_str(&format!("- **Centrality (PageRank)**: {:.4}\n", n.pagerank));

                if include_content {
                    if let Some(ref snip) = n.snippet {
                        let clean_snip = snip.trim();
                        if !clean_snip.is_empty() {
                            md.push_str("\n```markdown\n");
                            md.push_str(clean_snip);
                            md.push_str("\n```\n");
                        }
                    }
                }
                md.push('\n');
            }
        }

        md
    }
}
