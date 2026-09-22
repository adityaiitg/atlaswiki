use std::path::Path;
use tempfile::TempDir;

use atlaswiki_core::diagnostics::DiagnosticsEngine;
use atlaswiki_core::graph::KnowledgeGraph;
use atlaswiki_core::retrieval::{QueryClassifier, QueryIntent};
use atlaswiki_core::security::{MarkdownSanitizer, VaultRoot};
use atlaswiki_core::storage::StorageEngine;
use atlaswiki_parser::MarkdownParser;

#[test]
fn test_security_vault_root_containment() {
    let temp_dir = TempDir::new().unwrap();
    let root_path = temp_dir.path().canonicalize().unwrap();
    let vault_root = VaultRoot::new(&root_path).unwrap();

    // Normal safe file
    assert!(vault_root.lexical_sanitize_relative("Notes/Subfolder/NoteA.md").is_ok());

    // Path traversal attempt
    assert!(vault_root.lexical_sanitize_relative("../../../etc/passwd").is_err());
}

#[test]
fn test_markdown_sanitizer_xss_protection() {
    let dirty = "<script>alert('xss')</script>Hello **World**";
    let clean = MarkdownSanitizer::escape_html(dirty);
    assert!(!clean.contains("<script>"));
    assert!(clean.contains("&lt;script&gt;"));

    let safe_url = "https://example.com/page?query=1#anchor";
    assert!(MarkdownSanitizer::is_safe_url(safe_url));

    let bad_url = "javascript:alert(document.cookie)";
    assert!(!MarkdownSanitizer::is_safe_url(bad_url));
}

#[test]
fn test_storage_engine_document_lifecycle_and_fts() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("atlaswiki.db");
    let storage = StorageEngine::open(&db_path).unwrap();
    let parser = MarkdownParser::new();

    let doc_a_md = r#"---
title: Machine Learning
tags:
  - ai/ml
  - cs
aliases:
  - ML
---

# Machine Learning Overview

Machine learning is a subset of artificial intelligence. It focuses on algorithms that learn from data.
See also [[Deep Learning]] and [[Statistics#Bayes]].

## Applications

ML is used in computer vision, natural language processing, and robotics.
"#;

    let doc_b_md = r#"---
title: Deep Learning
tags:
  - ai/ml
---

# Deep Learning

Deep learning is part of machine learning based on artificial neural networks.
References [[Machine Learning]] for foundational theory.
"#;

    let parsed_a = parser.parse_file(Path::new("Machine Learning.md"), doc_a_md).unwrap();
    let parsed_b = parser.parse_file(Path::new("Deep Learning.md"), doc_b_md).unwrap();

    storage
        .sync_document(&parsed_a, "doc-1", "Machine Learning.md", 1000, 500, None)
        .unwrap();
    storage
        .sync_document(&parsed_b, "doc-2", "Deep Learning.md", 1001, 300, None)
        .unwrap();

    // Verify stats
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.total_documents, 2);
    assert!(stats.total_chunks >= 2);
    assert_eq!(stats.total_links, 3); // [[Deep Learning]], [[Statistics#Bayes]], [[Machine Learning]]

    // Verify FTS search (chunk_id, title, breadcrumbs, snippet, bm25_score)
    let fts_results = storage.search_fts("algorithms data learning", 10).unwrap();
    assert!(!fts_results.is_empty());
    assert_eq!(fts_results[0].1, "Machine Learning");

    // Verify Backlinks to Machine Learning
    let backlinks = storage.get_backlinks("Machine Learning").unwrap();
    assert_eq!(backlinks.len(), 1);
    assert_eq!(backlinks[0].source_path, "Deep Learning.md");

    // Verify Document Retrieval
    let doc_record = storage.get_document("doc-1").unwrap();
    assert!(doc_record.is_some());
    assert_eq!(doc_record.unwrap().title, "Machine Learning");

    let chunks = storage.get_chunks_for_doc("doc-1").unwrap();
    assert!(!chunks.is_empty());

    // Update document and check atomic replacement
    let doc_a_updated_md = r#"---
title: Machine Learning
tags:
  - ai/ml/updated
---

# Machine Learning Overview (Updated)

Updated text with algorithms and mathematical foundations.
"#;
    let parsed_a_updated = parser
        .parse_file(Path::new("Machine Learning.md"), doc_a_updated_md)
        .unwrap();
    storage
        .sync_document(&parsed_a_updated, "doc-1", "Machine Learning.md", 2000, 400, None)
        .unwrap();

    let updated_stats = storage.get_stats().unwrap();
    assert_eq!(updated_stats.total_documents, 2);
    assert_eq!(updated_stats.total_links, 1); // Only doc_b's link remains
}

#[test]
fn test_knowledge_graph_pagerank_and_shortest_path() {
    let parser = MarkdownParser::new();

    // Note A -> Note B -> Note C
    // Note D (orphan)
    let note_a = parser.parse_file(Path::new("A.md"), "# Note A\nConnects to [[Note B]].").unwrap();
    let note_b = parser.parse_file(Path::new("B.md"), "# Note B\nConnects to [[Note C]].").unwrap();
    let note_c = parser.parse_file(Path::new("C.md"), "# Note C\nEnd of chain. Mentions [[Unwritten Note]].").unwrap();
    let note_d = parser.parse_file(Path::new("D.md"), "# Note D\nStandalone orphan note.").unwrap();

    let mut graph = KnowledgeGraph::from_documents(&[note_a, note_b, note_c, note_d]);

    // Verify Shortest Path A -> C: ["Note A", "Note B", "Note C"]
    let path = graph.shortest_path("Note A", "Note C");
    assert!(path.is_some());
    let path_nodes = path.unwrap();
    assert_eq!(path_nodes, vec!["Note A", "Note B", "Note C"]);

    // Shortest path between disconnected nodes A and D
    let disconnected_path = graph.shortest_path("Note A", "Note D");
    assert!(disconnected_path.is_none());

    // Orphans: Note D has in-degree 0 and out-degree 0
    let orphans = graph.get_orphans();
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0].title, "Note D");

    // Wanted / Dangling pages: "Unwritten Note" is referenced by Note C
    let wanted = graph.get_wanted_pages();
    assert!(!wanted.is_empty());
    assert_eq!(wanted[0].0, "Unwritten Note");
    assert_eq!(wanted[0].1, 1);

    // PageRank calculation
    graph.compute_pagerank(0.85, 50);
    assert!(graph.get_pagerank("Note B") >= 0.0);
    assert!(graph.get_pagerank("Note C") >= 0.0);

    // D3 JSON export
    let d3_json = graph.to_d3_json();
    assert!(!d3_json.nodes.is_empty());
    assert!(!d3_json.links.is_empty());
}

#[test]
fn test_diagnostics_engine_broken_links_and_typos() {
    let parser = MarkdownParser::new();
    let mut diagnostics = DiagnosticsEngine::new();

    let note_ml = parser.parse_file(
        Path::new("Machine Learning.md"),
        "---\ntitle: Machine Learning\naliases: [ML]\n---\n# Overview\n\nContent here.\n\n# Algorithms\n\nSupervised learning.\n^algo-block\n",
    ).unwrap();

    let note_query = parser.parse_file(
        Path::new("AI Project.md"),
        "# AI Project\n\nUsing [[Machine Learnin#Overview]] and [[Machine Learning#NonExistentHeading]] and [[Machine Learning#^algo-block]] and [[Machine Learning#^bad-block]].\nAlso references [[Ghost Note]].\n",
    ).unwrap();

    diagnostics.index_document(&note_ml);
    diagnostics.index_document(&note_query);

    let report = diagnostics.run(&[note_query], false);
    assert!(!report.diagnostics.is_empty());

    // 1. Ghost Note is unresolved
    let ghost = report.diagnostics.iter().find(|d| d.message.contains("Ghost Note"));
    assert!(ghost.is_some());

    // 2. Machine Learnin is a typo for Machine Learning
    let typo = report.diagnostics.iter().find(|d| d.message.contains("Machine Learnin"));
    assert!(typo.is_some());
    assert!(typo.unwrap().suggestion.is_some());
    assert_eq!(typo.unwrap().suggestion.as_deref(), Some("[[Machine Learning]]"));

    // 3. NonExistentHeading is a broken heading
    let bad_heading = report.diagnostics.iter().find(|d| d.message.contains("NonExistentHeading"));
    assert!(bad_heading.is_some());

    // 4. bad-block is a broken block id
    let bad_block = report.diagnostics.iter().find(|d| d.message.contains("bad-block"));
    assert!(bad_block.is_some());
}

#[test]
fn test_retrieval_query_classification() {
    assert_eq!(QueryClassifier::classify("fn compute_hash()").intent, QueryIntent::CodeSymbol);
    assert_eq!(QueryClassifier::classify("struct VaultConfig").intent, QueryIntent::CodeSymbol);
    assert_eq!(QueryClassifier::classify("what is the main idea behind page rank in graphs?").intent, QueryIntent::NaturalLanguage);
    assert_eq!(QueryClassifier::classify("hybrid search algorithm").intent, QueryIntent::BalancedHybrid);
}
