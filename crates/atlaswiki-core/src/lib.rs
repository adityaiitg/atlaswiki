//! AtlasWiki Core Engine.
//!
//! Provides relational storage (SQLite WAL + FTS5), knowledge graph (Petgraph),
//! Model2Vec static vector embeddings, hybrid multi-signal retrieval, real-time watcher,
//! link diagnostics, and security boundary enforcement.

pub mod diagnostics;
pub mod embedder;
pub mod graph;
pub mod graph_rag;
pub mod hnsw;
pub mod retrieval;
pub mod security;
pub mod storage;
pub mod synthesis;
pub mod topology;
pub mod watcher;

pub use diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticsEngine, DiagnosticsReport,
    WantedPage,
};
pub use embedder::{EmbeddingEngine, Model2VecEmbedder};
pub use graph::{D3GraphData, EdgeType, KnowledgeGraph, LinkEdge, NoteNode};
pub use hnsw::{
    BenchmarkReport, BruteForceScan, HnswConfig, HnswIndex, Sq8Vector, SweepPoint, VectorEvalSuite,
};
pub use retrieval::{HybridRetriever, QueryClassifier, QueryIntent, SearchResult};
pub use security::{DosGuards, MarkdownSanitizer, VaultRoot};
pub use storage::{BacklinkRecord, StorageEngine, UnresolvedLinkRecord, VaultStats};
pub use synthesis::{
    update_synthesis_content, LivingWikiReport, MocReport, MocSynthesizer, NoteSummary,
    TagTaxonomy, TopicMocInfo, SYNTHESIS_BEGIN, SYNTHESIS_END,
};
pub use watcher::{RenameReport, VaultSyncEvent, VaultWatcher, VaultWatcherConfig};

