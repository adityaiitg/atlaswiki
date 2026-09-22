use std::path::Path;
use atlaswiki_parser::{LinkType, MarkdownParser};

#[test]
fn test_frontmatter_and_title() {
    let markdown = r#"---
title: "Quantum Computing Foundations"
aliases:
  - QC
  - Quantum
tags:
  - physics
  - computer-science
created: 2026-09-22
status: active
---

# Introduction to Qubits

A qubit is the basic unit of quantum information.
"#;

    let parser = MarkdownParser::new();
    let doc = parser
        .parse_file(Path::new("quantum.md"), markdown)
        .unwrap();

    assert_eq!(doc.title, "Quantum Computing Foundations");
    assert_eq!(doc.frontmatter.aliases, vec!["QC", "Quantum"]);
    assert_eq!(doc.frontmatter.tags, vec!["physics", "computer-science"]);
    assert_eq!(
        doc.frontmatter.extra.get("status").unwrap(),
        &serde_json::json!("active")
    );
}

#[test]
fn test_hierarchical_sections_and_breadcrumbs() {
    let markdown = r#"# Architecture

High-level architecture overview.

## Storage Layer

Details about SQLite and persistence.

### FTS5 Search

Full text search index details.

## Graph Engine

Petgraph knowledge graph.
"#;

    let parser = MarkdownParser::new();
    let doc = parser
        .parse_file(Path::new("arch.md"), markdown)
        .unwrap();

    assert_eq!(doc.sections.len(), 4);
    assert_eq!(doc.sections[0].heading, "Architecture");
    assert_eq!(doc.sections[0].level, 1);
    assert_eq!(doc.sections[0].breadcrumbs, vec!["Architecture"]);

    assert_eq!(doc.sections[1].heading, "Storage Layer");
    assert_eq!(doc.sections[1].level, 2);
    assert_eq!(
        doc.sections[1].breadcrumbs,
        vec!["Architecture", "Storage Layer"]
    );

    assert_eq!(doc.sections[2].heading, "FTS5 Search");
    assert_eq!(doc.sections[2].level, 3);
    assert_eq!(
        doc.sections[2].breadcrumbs,
        vec!["Architecture", "Storage Layer", "FTS5 Search"]
    );

    assert_eq!(doc.sections[3].heading, "Graph Engine");
    assert_eq!(doc.sections[3].level, 2);
    assert_eq!(
        doc.sections[3].breadcrumbs,
        vec!["Architecture", "Graph Engine"]
    );
}

#[test]
fn test_wikilinks_and_embeds() {
    let markdown = r#"# Notes

Reference to [[Machine Learning]] and alias [[Artificial Intelligence|AI]].
Also target with heading [[Neural Networks#Backpropagation]].
And embedded diagram ![[Flowchart Diagram]].
"#;

    let parser = MarkdownParser::new();
    let doc = parser
        .parse_file(Path::new("notes.md"), markdown)
        .unwrap();

    assert_eq!(doc.links.len(), 4);

    assert_eq!(doc.links[0].link_type, LinkType::Wikilink);
    assert_eq!(doc.links[0].target_note, "Machine Learning");
    assert_eq!(doc.links[0].alias, None);

    assert_eq!(doc.links[1].link_type, LinkType::Wikilink);
    assert_eq!(doc.links[1].target_note, "Artificial Intelligence");
    assert_eq!(doc.links[1].alias, Some("AI".to_string()));

    assert_eq!(doc.links[2].link_type, LinkType::Wikilink);
    assert_eq!(doc.links[2].target_note, "Neural Networks");
    assert_eq!(
        doc.links[2].target_heading,
        Some("Backpropagation".to_string())
    );

    assert_eq!(doc.links[3].link_type, LinkType::Embed);
    assert_eq!(doc.links[3].target_note, "Flowchart Diagram");
}

#[test]
fn test_code_blocks_ignore_links_and_tags() {
    let markdown = r#"# Code Sample

Here is legitimate text with #valid-tag and [[ValidLink]].

```python
# This is a python comment, NOT a tag: #fake-tag
def test():
    # And this is NOT a wikilink: [[FakeLink]]
    return True
```

Back to normal text with #second-tag.
"#;

    let parser = MarkdownParser::new();
    let doc = parser
        .parse_file(Path::new("code.md"), markdown)
        .unwrap();

    // Only ValidLink should be captured
    assert_eq!(doc.links.len(), 1);
    assert_eq!(doc.links[0].target_note, "ValidLink");

    // Only valid-tag and second-tag should be captured
    let tag_names: Vec<String> = doc.tags.into_iter().map(|t| t.name).collect();
    assert!(tag_names.contains(&"valid-tag".to_string()));
    assert!(tag_names.contains(&"second-tag".to_string()));
    assert!(!tag_names.contains(&"fake-tag".to_string()));
}

#[test]
fn test_block_references_and_standard_links() {
    let markdown = r#"# Block Ref Test

Here is a block reference link [[MyNote#^block-abc|Custom Label]].
Also a standard markdown link [Relative Note](subfolder/target.md).
"#;

    let parser = MarkdownParser::new();
    let doc = parser
        .parse_file(Path::new("block.md"), markdown)
        .unwrap();

    assert_eq!(doc.links.len(), 2);
    assert_eq!(doc.links[0].target_note, "MyNote");
    assert_eq!(doc.links[0].target_block, Some("block-abc".to_string()));
    assert_eq!(doc.links[0].alias, Some("Custom Label".to_string()));

    assert_eq!(doc.links[1].link_type, LinkType::Markdown);
    assert_eq!(doc.links[1].target_note, "subfolder/target");
    assert_eq!(doc.links[1].alias, Some("Relative Note".to_string()));
}

#[test]
fn test_semantic_chunks_carry_breadcrumbs() {
    let markdown = r#"# Deep Learning

Overview of deep learning.

## Computer Vision

Vision models and architectures.

### Convolutional Networks

CNNs revolutionized image processing with local receptive fields.
"#;

    let parser = MarkdownParser::new();
    let doc = parser.parse_file(Path::new("dl.md"), markdown).unwrap();

    assert!(doc.chunks.len() >= 3);
    let cnn_chunk = doc
        .chunks
        .iter()
        .find(|c| c.breadcrumbs.contains("Convolutional Networks"))
        .unwrap();

    assert_eq!(
        cnn_chunk.breadcrumbs,
        "Deep Learning > Computer Vision > Convolutional Networks"
    );
    assert!(cnn_chunk.content.contains("CNNs revolutionized"));
}
