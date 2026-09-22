//! Knowledge Graph & Petgraph Architecture for AtlasWiki.
//!
//! Provides bidirectional graph modeling, PageRank centrality calculation,
//! shortest path traversal, orphan/wanted page diagnostics, and D3.js JSON serialization.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use atlaswiki_parser::ast::{LinkType, ParsedDocument};

/// Type of node in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Document,
    Dangling, // Unresolved note reference
    Tag,
}

/// Metadata stored inside each node in the Petgraph graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteNode {
    pub id: String,
    pub title: String,
    pub path: Option<PathBuf>,
    pub node_type: NodeType,
    pub tags: Vec<String>,
    pub word_count: usize,
    pub pagerank: f32,
}

/// Edge types connecting nodes in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    Wikilink,
    Embed,
    Markdown,
    TaggedWith,
}

/// Metadata stored on each directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkEdge {
    pub edge_type: EdgeType,
    pub heading: Option<String>,
    pub block: Option<String>,
    pub alias: Option<String>,
    pub line_number: usize,
    pub context_snippet: Option<String>,
    pub weight: f32,
}

/// A node formatted for D3.js force-directed visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Node {
    pub id: String,
    pub title: String,
    pub group: String,
    pub pagerank: f32,
    pub in_degree: usize,
    pub out_degree: usize,
    pub is_dangling: bool,
}

/// A link formatted for D3.js force-directed visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Link {
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub weight: f32,
}

/// Graph export data payload for the interactive web viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3GraphData {
    pub nodes: Vec<D3Node>,
    pub links: Vec<D3Link>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Core Knowledge Graph engine wrapping Petgraph's `DiGraph`.
pub struct KnowledgeGraph {
    graph: DiGraph<NoteNode, LinkEdge>,
    node_map: HashMap<String, NodeIndex>,      // canonical id -> NodeIndex
    title_map: HashMap<String, NodeIndex>,     // lowercase title -> NodeIndex
    alias_map: HashMap<String, NodeIndex>,     // lowercase alias -> NodeIndex
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            title_map: HashMap::new(),
            alias_map: HashMap::new(),
        }
    }

    /// Construct knowledge graph from an array of parsed documents.
    pub fn from_documents(docs: &[ParsedDocument]) -> Self {
        let mut kg = Self::new();

        // Pass 1: Insert all document nodes
        for doc in docs {
            let doc_id = doc.title.clone();
            let tags: Vec<String> = doc.tags.iter().map(|t| t.name.clone()).collect();

            let node = NoteNode {
                id: doc_id.clone(),
                title: doc.title.clone(),
                path: Some(doc.path.clone()),
                node_type: NodeType::Document,
                tags,
                word_count: doc.word_count,
                pagerank: 0.0,
            };

            let idx = kg.graph.add_node(node);
            kg.node_map.insert(doc_id.clone(), idx);
            kg.title_map.insert(doc.title.to_lowercase(), idx);

            for alias in &doc.frontmatter.aliases {
                kg.alias_map.insert(alias.to_lowercase(), idx);
            }
        }

        // Pass 2: Insert edges (wikilinks, embeds, standard links)
        for doc in docs {
            let source_idx = match kg.node_map.get(&doc.title) {
                Some(&idx) => idx,
                None => continue,
            };

            for link in &doc.links {
                let target_idx = kg.resolve_or_create_dangling(&link.target_note);

                let (edge_type, weight) = match link.link_type {
                    LinkType::Embed => (EdgeType::Embed, 2.0),
                    LinkType::Wikilink => (EdgeType::Wikilink, 1.0),
                    LinkType::Markdown => (EdgeType::Markdown, 0.8),
                };

                kg.graph.add_edge(
                    source_idx,
                    target_idx,
                    LinkEdge {
                        edge_type,
                        heading: link.target_heading.clone(),
                        block: link.target_block.clone(),
                        alias: link.alias.clone(),
                        line_number: link.line_number,
                        context_snippet: link.context_snippet.clone(),
                        weight,
                    },
                );
            }
        }

        // Pass 3: Compute PageRank centrality
        kg.compute_pagerank(0.85, 30);

        kg
    }

    /// Resolves target note to NodeIndex, or creates a Dangling note node.
    fn resolve_or_create_dangling(&mut self, target_note: &str) -> NodeIndex {
        let lower = target_note.to_lowercase();

        if let Some(&idx) = self.title_map.get(&lower) {
            return idx;
        }
        if let Some(&idx) = self.alias_map.get(&lower) {
            return idx;
        }
        if let Some(&idx) = self.node_map.get(target_note) {
            return idx;
        }

        // Create Dangling node
        let dangling_node = NoteNode {
            id: target_note.to_string(),
            title: target_note.to_string(),
            path: None,
            node_type: NodeType::Dangling,
            tags: Vec::new(),
            word_count: 0,
            pagerank: 0.0,
        };

        let idx = self.graph.add_node(dangling_node);
        self.node_map.insert(target_note.to_string(), idx);
        self.title_map.insert(lower, idx);
        idx
    }

    /// Computes Dangling-Safe PageRank using power iteration.
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

        // Normalize PageRank scores and assign to nodes
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

    /// Returns the normalized PageRank score for a given note title (0.0 to 1.0).
    pub fn get_pagerank(&self, title: &str) -> f32 {
        let lower = title.to_lowercase();
        if let Some(&idx) = self.title_map.get(&lower) {
            return self.graph[idx].pagerank;
        }
        0.0
    }

    /// Returns all incoming backlinks to a given note.
    pub fn get_backlinks(&self, note_title: &str) -> Vec<(NoteNode, LinkEdge)> {
        let lower = note_title.to_lowercase();
        let target_idx = match self.title_map.get(&lower).or_else(|| self.alias_map.get(&lower)) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let mut backlinks = Vec::new();
        for edge in self.graph.edges_directed(target_idx, Direction::Incoming) {
            let source_idx = edge.source();
            let source_node = self.graph[source_idx].clone();
            let edge_data = edge.weight().clone();
            backlinks.push((source_node, edge_data));
        }

        backlinks
    }

    /// Finds the shortest path between Note A and Note B using bidirectional BFS.
    pub fn shortest_path(&self, from_title: &str, to_title: &str) -> Option<Vec<String>> {
        let from_idx = *self.title_map.get(&from_title.to_lowercase())?;
        let to_idx = *self.title_map.get(&to_title.to_lowercase())?;

        if from_idx == to_idx {
            return Some(vec![self.graph[from_idx].title.clone()]);
        }

        let mut forward_queue = VecDeque::new();
        let mut backward_queue = VecDeque::new();

        let mut forward_parents: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut backward_parents: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        forward_queue.push_back(from_idx);
        backward_queue.push_back(to_idx);

        let mut meeting_node: Option<NodeIndex> = None;

        while !forward_queue.is_empty() && !backward_queue.is_empty() {
            // Forward step
            if let Some(curr) = forward_queue.pop_front() {
                for neighbor in self.graph.neighbors_directed(curr, Direction::Outgoing) {
                    if !forward_parents.contains_key(&neighbor) && neighbor != from_idx {
                        forward_parents.insert(neighbor, curr);
                        forward_queue.push_back(neighbor);

                        if backward_parents.contains_key(&neighbor) || neighbor == to_idx {
                            meeting_node = Some(neighbor);
                            break;
                        }
                    }
                }
            }

            if meeting_node.is_some() {
                break;
            }

            // Backward step
            if let Some(curr) = backward_queue.pop_front() {
                for neighbor in self.graph.neighbors_directed(curr, Direction::Incoming) {
                    if !backward_parents.contains_key(&neighbor) && neighbor != to_idx {
                        backward_parents.insert(neighbor, curr);
                        backward_queue.push_back(neighbor);

                        if forward_parents.contains_key(&neighbor) || neighbor == from_idx {
                            meeting_node = Some(neighbor);
                            break;
                        }
                    }
                }
            }

            if meeting_node.is_some() {
                break;
            }
        }

        let meet = meeting_node?;

        // Reconstruct path: from -> meet -> to
        let mut path_from = Vec::new();
        let mut curr = meet;
        while curr != from_idx {
            path_from.push(self.graph[curr].title.clone());
            curr = *forward_parents.get(&curr)?;
        }
        path_from.push(self.graph[from_idx].title.clone());
        path_from.reverse();

        let mut curr = meet;
        while curr != to_idx {
            if let Some(&next) = backward_parents.get(&curr) {
                path_from.push(self.graph[next].title.clone());
                curr = next;
            } else {
                break;
            }
        }

        Some(path_from)
    }

    /// Detect isolated orphan notes (in-degree = 0 and out-degree = 0).
    pub fn get_orphans(&self) -> Vec<NoteNode> {
        let mut orphans = Vec::new();
        for idx in self.graph.node_indices() {
            let in_deg = self.graph.neighbors_directed(idx, Direction::Incoming).count();
            let out_deg = self.graph.neighbors_directed(idx, Direction::Outgoing).count();

            if in_deg == 0 && out_deg == 0 && self.graph[idx].node_type == NodeType::Document {
                orphans.push(self.graph[idx].clone());
            }
        }
        orphans
    }

    /// List wanted/dangling pages sorted by incoming reference count.
    pub fn get_wanted_pages(&self) -> Vec<(String, usize)> {
        let mut wanted = Vec::new();
        for idx in self.graph.node_indices() {
            if self.graph[idx].node_type == NodeType::Dangling {
                let in_deg = self.graph.neighbors_directed(idx, Direction::Incoming).count();
                wanted.push((self.graph[idx].title.clone(), in_deg));
            }
        }
        wanted.sort_by_key(|a| std::cmp::Reverse(a.1));
        wanted
    }

    /// Serialize complete knowledge graph for D3.js interactive visualization.
    pub fn to_d3_json(&self) -> D3GraphData {
        let mut nodes = Vec::new();
        let mut links = Vec::new();

        for idx in self.graph.node_indices() {
            let n = &self.graph[idx];
            let in_deg = self.graph.neighbors_directed(idx, Direction::Incoming).count();
            let out_deg = self.graph.neighbors_directed(idx, Direction::Outgoing).count();

            let group = if !n.tags.is_empty() {
                n.tags[0].clone()
            } else if n.node_type == NodeType::Dangling {
                "dangling".to_string()
            } else {
                "default".to_string()
            };

            nodes.push(D3Node {
                id: n.id.clone(),
                title: n.title.clone(),
                group,
                pagerank: n.pagerank,
                in_degree: in_deg,
                out_degree: out_deg,
                is_dangling: n.node_type == NodeType::Dangling,
            });
        }

        for edge in self.graph.edge_references() {
            let source_idx = edge.source();
            let target_idx = edge.target();

            let edge_type_str = match edge.weight().edge_type {
                EdgeType::Wikilink => "wikilink",
                EdgeType::Embed => "embed",
                EdgeType::Markdown => "markdown",
                EdgeType::TaggedWith => "tagged_with",
            };

            links.push(D3Link {
                source: self.graph[source_idx].id.clone(),
                target: self.graph[target_idx].id.clone(),
                edge_type: edge_type_str.to_string(),
                weight: edge.weight().weight,
            });
        }

        let total_nodes = nodes.len();
        let total_edges = links.len();

        D3GraphData {
            nodes,
            links,
            total_nodes,
            total_edges,
        }
    }
}
