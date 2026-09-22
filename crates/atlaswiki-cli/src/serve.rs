//! Embedded HTTP web server for AtlasWiki interactive knowledge graph visualizer.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;
use tiny_http::{Header, Response, Server, StatusCode};

use atlaswiki_core::graph::KnowledgeGraph;
use atlaswiki_core::storage::StorageEngine;
use atlaswiki_parser::MarkdownParser;

pub const VIEWER_HTML: &str = include_str!("../assets/viewer.html");

pub struct WebServer {
    port: u16,
    vault_path: PathBuf,
    storage: Arc<StorageEngine>,
}

impl WebServer {
    pub fn new<P: AsRef<Path>>(vault_path: P, storage: Arc<StorageEngine>, port: u16) -> Self {
        Self {
            port,
            vault_path: vault_path.as_ref().to_path_buf(),
            storage,
        }
    }

    pub fn run(&self) -> Result<()> {
        let addr = format!("127.0.0.1:{}", self.port);
        let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

        println!(
            "{} AtlasWiki graph visualizer running at {}",
            "✓".green().bold(),
            format!("http://{}", addr).cyan().underline().bold()
        );
        println!("{}", "Press Ctrl+C to stop the server.\n".dimmed());

        for request in server.incoming_requests() {
            let url = request.url().to_string();
            let (path, query) = match url.split_once('?') {
                Some((p, q)) => (p, q),
                None => (url.as_str(), ""),
            };

            let response = match path {
                "/" | "/index.html" => {
                    let mut resp = Response::from_string(VIEWER_HTML);
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
                    resp
                }
                "/api/graph" => {
                    // Rebuild graph from database documents
                    let graph = self.build_graph();
                    let d3_data = graph.to_d3_json();
                    let json_str = serde_json::to_string(&d3_data).unwrap_or_else(|_| "{}".to_string());
                    let mut resp = Response::from_string(json_str);
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    resp
                }
                "/api/stats" => {
                    let stats = self.storage.get_stats().unwrap_or(atlaswiki_core::storage::VaultStats {
                        total_documents: 0,
                        total_sections: 0,
                        total_links: 0,
                        total_tags: 0,
                        total_chunks: 0,
                        total_embeddings: 0,
                        total_words: 0,
                    });
                    let json_str = serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string());
                    let mut resp = Response::from_string(json_str);
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp
                }
                "/api/backlinks" => {
                    let title = Self::extract_param(query, "title").unwrap_or_default();
                    let backlinks = self.storage.get_backlinks(&title).unwrap_or_default();
                    let json_str = serde_json::to_string(&backlinks).unwrap_or_else(|_| "[]".to_string());
                    let mut resp = Response::from_string(json_str);
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp
                }
                "/api/note" => {
                    let title = Self::extract_param(query, "title").unwrap_or_default();
                    if let Ok(Some(doc)) = self.storage.get_document(&title) {
                        let full_path = self.vault_path.join(&doc.path);
                        let content = std::fs::read_to_string(&full_path).unwrap_or_default();
                        let payload = serde_json::json!({
                            "title": doc.title,
                            "path": doc.path,
                            "word_count": doc.word_count,
                            "content": content
                        });
                        let mut resp = Response::from_string(payload.to_string());
                        resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                        resp
                    } else {
                        Response::from_string(r#"{"error":"Note not found"}"#)
                            .with_status_code(StatusCode(404))
                    }
                }
                "/api/search" => {
                    let q = Self::extract_param(query, "q").unwrap_or_default();
                    let limit = Self::extract_param(query, "limit")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(20);

                    let results = self.storage.search_fts(&q, limit).unwrap_or_default();
                    let hits: Vec<serde_json::Value> = results
                        .into_iter()
                        .map(|(chunk_id, title, breadcrumbs, snippet, score)| {
                            serde_json::json!({
                                "chunk_id": chunk_id,
                                "title": title,
                                "breadcrumbs": breadcrumbs,
                                "snippet": snippet,
                                "score": score
                            })
                        })
                        .collect();

                    let json_str = serde_json::to_string(&hits).unwrap_or_else(|_| "[]".to_string());
                    let mut resp = Response::from_string(json_str);
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp
                }
                _ => Response::from_string("Not Found").with_status_code(StatusCode(404)),
            };

            let _ = request.respond(response);
        }

        Ok(())
    }

    fn extract_param(query: &str, key: &str) -> Option<String> {
        for pair in query.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                if k == key {
                    return urlencoding_decode(v);
                }
            }
        }
        None
    }

    fn build_graph(&self) -> KnowledgeGraph {
        let parser = MarkdownParser::new();
        let paths = self.storage.get_all_document_paths().unwrap_or_default();
        let mut docs = Vec::new();

        for p in paths {
            let full_p = self.vault_path.join(&p);
            if let Ok(content) = std::fs::read_to_string(&full_p) {
                if let Ok(doc) = parser.parse_file(&full_p, &content) {
                    docs.push(doc);
                }
            }
        }

        let mut kg = KnowledgeGraph::from_documents(&docs);
        kg.compute_pagerank(0.85, 30);
        kg
    }
}

fn urlencoding_decode(input: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next()?.to_digit(16)? as u8;
            let h2 = chars.next()?.to_digit(16)? as u8;
            bytes.push((h1 << 4) | h2);
        } else if c == '+' {
            bytes.push(b' ');
        } else {
            bytes.push(c as u8);
        }
    }

    String::from_utf8(bytes).ok()
}
