use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub path: PathBuf,
    pub title: String,
    pub frontmatter: Frontmatter,
    pub sections: Vec<AstSection>,
    pub links: Vec<AstLink>,
    pub tags: Vec<AstTag>,
    pub chunks: Vec<AstChunk>,
    pub word_count: usize,
    pub content_hash: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Frontmatter {
    pub title: Option<String>,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub created: Option<String>,
    pub updated: Option<String>,
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstSection {
    pub id: String,
    pub heading: String,
    pub level: u8,
    pub parent_id: Option<String>,
    pub breadcrumbs: Vec<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkType {
    Wikilink,
    Embed,
    Markdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstLink {
    pub link_type: LinkType,
    pub target_note: String,
    pub target_heading: Option<String>,
    pub target_block: Option<String>,
    pub alias: Option<String>,
    pub line_number: usize,
    pub context_snippet: Option<String>,
    pub source_section_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstTag {
    pub name: String,
    pub line_number: usize,
    pub section_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstChunk {
    pub chunk_id: String,
    pub section_id: Option<String>,
    pub title: String,
    pub breadcrumbs: String,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
}
