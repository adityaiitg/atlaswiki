//! Language Server Protocol (LSP) 3.17 stdio server for AtlasWiki.
//!
//! Exposes:
//! - textDocument/completion: note titles, frontmatter aliases, and tags
//! - textDocument/definition: jump to exact file path and heading line position
//! - textDocument/hover: note preview, transclusions, and section excerpt

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use atlaswiki_core::storage::StorageEngine;
use atlaswiki_parser::ast::Frontmatter;

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CachedNote {
    pub doc_id: String,
    pub title: String,
    pub path: String,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    pub word_count: usize,
}

pub struct LspServer {
    pub vault_root: PathBuf,
    pub storage: Arc<StorageEngine>,
    pub notes_cache: Vec<CachedNote>,
    pub tags_cache: Vec<String>,
    pub open_documents: HashMap<String, String>,
    wikilink_re: Regex,
}

impl LspServer {
    pub fn new(vault_root: PathBuf, storage: Arc<StorageEngine>) -> Self {
        let wikilink_re = Regex::new(
            r"(?P<embed>!)?\[\[(?P<target>[^\]|#^]+)(?:#(?:(?:\^(?P<block>[^\]|]+))|(?P<heading>[^\]|^|]+)(?:\^(?P<block2>[^\]|]+))?))?(?:\^(?P<block_direct>[^\]|]+))?(?:\|(?P<alias>[^\]]+))?\]\]"
        ).expect("valid wikilink regex");

        let mut server = Self {
            vault_root,
            storage,
            notes_cache: Vec::new(),
            tags_cache: Vec::new(),
            open_documents: HashMap::new(),
            wikilink_re,
        };
        server.reload_cache();
        server
    }

    pub fn reload_cache(&mut self) {
        if let Ok(docs) = self.storage.get_all_documents() {
            let mut cached = Vec::with_capacity(docs.len());
            for d in docs {
                let fm: Frontmatter = serde_json::from_str(&d.frontmatter_json).unwrap_or_default();
                cached.push(CachedNote {
                    doc_id: d.doc_id,
                    title: d.title,
                    path: d.path,
                    aliases: fm.aliases,
                    tags: fm.tags,
                    word_count: d.word_count,
                });
            }
            self.notes_cache = cached;
        }

        if let Ok(tags) = self.storage.get_all_tags() {
            self.tags_cache = tags;
        }
    }

    fn get_doc_text(&self, uri: &str) -> Option<String> {
        if let Some(text) = self.open_documents.get(uri) {
            return Some(text.clone());
        }
        let file_path = uri_to_path(uri)?;
        fs::read_to_string(&file_path).ok()
    }

    pub fn handle_initialize(&mut self, params: &Value) -> Value {
        // Update vault_root from rootUri or rootPath if available
        if let Some(root_uri) = params.get("rootUri").and_then(|v| v.as_str()) {
            if let Some(path) = uri_to_path(root_uri) {
                if path.exists() {
                    self.vault_root = path;
                }
            }
        } else if let Some(root_path) = params.get("rootPath").and_then(|v| v.as_str()) {
            let path = PathBuf::from(root_path);
            if path.exists() {
                self.vault_root = path;
            }
        }

        self.reload_cache();

        json!({
            "capabilities": {
                "textDocumentSync": 1, // Full document sync
                "completionProvider": {
                    "triggerCharacters": ["[", "#"],
                    "resolveProvider": false
                },
                "definitionProvider": true,
                "hoverProvider": true
            },
            "serverInfo": {
                "name": "atlaswiki-lsp",
                "version": "0.1.0"
            }
        })
    }

    pub fn handle_completion(&self, params: &Value) -> Value {
        let uri = params
            .get("textDocument")
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or_default();

        let line = params
            .get("position")
            .and_then(|p| p.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as usize;

        let character = params
            .get("position")
            .and_then(|p| p.get("character"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        let trigger_char = params
            .get("context")
            .and_then(|ctx| ctx.get("triggerCharacter"))
            .and_then(|tc| tc.as_str());

        let doc_text = self.get_doc_text(uri).unwrap_or_default();
        let lines: Vec<&str> = doc_text.lines().collect();

        let line_content = lines.get(line).copied().unwrap_or("");
        let col = character.min(line_content.len());
        let before_cursor = &line_content[..col];

        let is_wikilink_trigger = trigger_char == Some("[")
            || before_cursor.ends_with("[[")
            || (before_cursor.rfind("[[").is_some()
                && before_cursor.rfind("[[").unwrap() > before_cursor.rfind("]]").unwrap_or(0));

        let is_tag_trigger = trigger_char == Some("#")
            || before_cursor.ends_with('#')
            || (before_cursor.rfind('#').is_some()
                && before_cursor[before_cursor.rfind('#').unwrap()..]
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '/' || c == '_' || c == '-' || c == '#'));

        let mut items = Vec::new();

        if is_wikilink_trigger {
            // Note completions
            for note in &self.notes_cache {
                items.push(json!({
                    "label": note.title,
                    "kind": 18, // Reference
                    "detail": note.path,
                    "insertText": note.title,
                    "documentation": {
                        "kind": "markdown",
                        "value": format!("**{}** ({} words)\n\nTags: {}", note.title, note.word_count, note.tags.join(", "))
                    }
                }));

                // Aliases
                for alias in &note.aliases {
                    items.push(json!({
                        "label": alias,
                        "kind": 18, // Reference
                        "detail": format!("alias of [[{}]]", note.title),
                        "insertText": format!("{}|{}", note.title, alias),
                        "filterText": alias,
                        "documentation": {
                            "kind": "markdown",
                            "value": format!("Alias for note **[[{}]]** (`{}`)", note.title, note.path)
                        }
                    }));
                }
            }
        } else if is_tag_trigger {
            for tag in &self.tags_cache {
                let clean_tag = tag.trim_start_matches('#');
                items.push(json!({
                    "label": format!("#{clean_tag}"),
                    "kind": 14, // Keyword
                    "detail": "tag",
                    "insertText": clean_tag,
                    "filterText": clean_tag,
                    "documentation": {
                        "kind": "markdown",
                        "value": format!("Vault tag `#{clean_tag}`")
                    }
                }));
            }
        } else {
            // Return all notes if triggered without explicit symbol
            for note in &self.notes_cache {
                items.push(json!({
                    "label": note.title,
                    "kind": 18,
                    "detail": note.path,
                    "insertText": note.title,
                }));
            }
        }

        json!({
            "isIncomplete": false,
            "items": items
        })
    }

    pub fn handle_definition(&self, params: &Value) -> Value {
        let uri = params
            .get("textDocument")
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or_default();

        let line = params
            .get("position")
            .and_then(|p| p.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as usize;

        let character = params
            .get("position")
            .and_then(|p| p.get("character"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        let doc_text = self.get_doc_text(uri).unwrap_or_default();
        let lines: Vec<&str> = doc_text.lines().collect();
        let line_content = lines.get(line).copied().unwrap_or("");

        // Find wikilink under cursor
        for cap in self.wikilink_re.captures_iter(line_content) {
            let full_match = cap.get(0).unwrap();
            let start = full_match.start();
            let end = full_match.end();

            if character >= start && character <= end {
                let target = cap.name("target").map(|m| m.as_str().trim()).unwrap_or("");
                let heading = cap.name("heading").map(|m| m.as_str().trim());

                if let Ok(Some(doc)) = self.storage.get_document(target) {
                    let target_path = self.vault_root.join(&doc.path);
                    let mut target_line: usize = 0;

                    if let Some(target_heading) = heading {
                        if let Ok(sections) = self.storage.get_sections_for_doc(&doc.doc_id) {
                            for sec in sections {
                                if sec.heading.eq_ignore_ascii_case(target_heading) {
                                    target_line = sec.line_start.saturating_sub(1);
                                    break;
                                }
                            }
                        }
                    }

                    let target_uri = path_to_uri(&target_path);
                    return json!({
                        "uri": target_uri,
                        "range": {
                            "start": { "line": target_line, "character": 0 },
                            "end": { "line": target_line, "character": 0 }
                        }
                    });
                }
            }
        }

        Value::Null
    }

    pub fn handle_hover(&self, params: &Value) -> Value {
        let uri = params
            .get("textDocument")
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or_default();

        let line = params
            .get("position")
            .and_then(|p| p.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as usize;

        let character = params
            .get("position")
            .and_then(|p| p.get("character"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;

        let doc_text = self.get_doc_text(uri).unwrap_or_default();
        let lines: Vec<&str> = doc_text.lines().collect();
        let line_content = lines.get(line).copied().unwrap_or("");

        for cap in self.wikilink_re.captures_iter(line_content) {
            let full_match = cap.get(0).unwrap();
            let start = full_match.start();
            let end = full_match.end();

            if character >= start && character <= end {
                let target = cap.name("target").map(|m| m.as_str().trim()).unwrap_or("");
                let heading = cap.name("heading").map(|m| m.as_str().trim());
                let is_embed = cap.name("embed").is_some();

                if let Ok(Some(doc)) = self.storage.get_document(target) {
                    if let Some(target_heading) = heading {
                        if let Ok(sections) = self.storage.get_sections_for_doc(&doc.doc_id) {
                            if let Some(sec) = sections.into_iter().find(|s| s.heading.eq_ignore_ascii_case(target_heading)) {
                                let badge = if is_embed { "Transclusion Embed" } else { "Section Link" };
                                let hover_text = format!(
                                    "### {} > {}\n*[{}] lines {}-{}*\n\n{}",
                                    doc.title, sec.heading, badge, sec.line_start, sec.line_end, sec.content.trim()
                                );
                                return json!({
                                    "contents": {
                                        "kind": "markdown",
                                        "value": hover_text
                                    }
                                });
                            }
                        }
                    }

                    // Full note hover preview
                    let target_path = self.vault_root.join(&doc.path);
                    let preview = match fs::read_to_string(&target_path) {
                        Ok(raw) => {
                            let body = if raw.starts_with("---") {
                                if let Some(end_idx) = raw[3..].find("---") {
                                    raw[3 + end_idx + 3..].trim()
                                } else {
                                    raw.as_str()
                                }
                            } else {
                                raw.as_str()
                            };
                            let truncated: String = body.chars().take(800).collect();
                            truncated
                        }
                        Err(_) => String::new(),
                    };

                    let badge = if is_embed { "Transclusion Note" } else { "Note Preview" };
                    let hover_text = format!(
                        "### {}\n*[{}] `{}`* | *Words: {}*\n\n---\n{}",
                        doc.title, badge, doc.path, doc.word_count, preview.trim()
                    );

                    return json!({
                        "contents": {
                            "kind": "markdown",
                            "value": hover_text
                        }
                    });
                }
            }
        }

        Value::Null
    }

    pub fn handle_did_open(&mut self, params: &Value) {
        if let Some(td) = params.get("textDocument") {
            if let (Some(uri), Some(text)) = (
                td.get("uri").and_then(|u| u.as_str()),
                td.get("text").and_then(|t| t.as_str()),
            ) {
                self.open_documents.insert(uri.to_string(), text.to_string());
            }
        }
    }

    pub fn handle_did_change(&mut self, params: &Value) {
        if let Some(uri) = params.get("textDocument").and_then(|td| td.get("uri")).and_then(|u| u.as_str()) {
            if let Some(changes) = params.get("contentChanges").and_then(|c| c.as_array()) {
                if let Some(last) = changes.last().and_then(|ch| ch.get("text")).and_then(|t| t.as_str()) {
                    self.open_documents.insert(uri.to_string(), last.to_string());
                }
            }
        }
    }

    pub fn handle_did_close(&mut self, params: &Value) {
        if let Some(uri) = params.get("textDocument").and_then(|td| td.get("uri")).and_then(|u| u.as_str()) {
            self.open_documents.remove(uri);
        }
    }
}

pub fn uri_to_path(uri: &str) -> Option<PathBuf> {
    if let Some(stripped) = uri.strip_prefix("file://") {
        let decoded = urlencoding_decode(stripped);
        Some(PathBuf::from(decoded))
    } else {
        Some(PathBuf::from(uri))
    }
}

pub fn path_to_uri(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let path_str = canonical.to_string_lossy();
    let encoded = urlencoding_encode(&path_str);
    format!("file://{encoded}")
}

fn urlencoding_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => {
                result.push(b as char);
            }
            b' ' => result.push_str("%20"),
            _ => {
                result.push_str(&format!("%{:02X}", b));
            }
        }
    }
    result
}

fn urlencoding_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(val) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                result.push(val as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

pub fn run_lsp_server(vault_root: &Path) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let db_path = canonical_vault.join(".atlaswiki").join("index.db");

    let storage = Arc::new(
        StorageEngine::open(&db_path)
            .with_context(|| format!("Failed to open index at {:?}", db_path))?,
    );

    let mut server = LspServer::new(canonical_vault, storage);

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    loop {
        // Read LSP HTTP headers
        let mut content_length: Option<usize> = None;
        let mut header_line = String::new();

        loop {
            header_line.clear();
            let bytes_read = reader.read_line(&mut header_line)?;
            if bytes_read == 0 {
                // EOF on stdin: client disconnected
                return Ok(());
            }

            let trimmed = header_line.trim_end_matches(|c| c == '\r' || c == '\n');
            if trimmed.is_empty() {
                // End of header section
                break;
            }

            if let Some(idx) = trimmed.find(':') {
                let name = trimmed[..idx].trim();
                let val = trimmed[idx + 1..].trim();
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = val.parse::<usize>().ok();
                }
            }
        }

        let length = match content_length {
            Some(l) => l,
            None => continue,
        };

        // Read exact message body
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body)?;

        let body_str = match std::str::from_utf8(&body) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let req: JsonRpcRequest = match serde_json::from_str(body_str) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                        data: None,
                    }),
                };
                send_response(&mut writer, &err_resp)?;
                continue;
            }
        };

        // Dispatch method
        match req.method.as_str() {
            "initialize" => {
                let result = server.handle_initialize(&req.params);
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(result),
                        error: None,
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
            "initialized" => {
                // Client initialized notification
            }
            "shutdown" => {
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(Value::Null),
                        error: None,
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
            "exit" => {
                std::process::exit(0);
            }
            "textDocument/didOpen" => {
                server.handle_did_open(&req.params);
            }
            "textDocument/didChange" => {
                server.handle_did_change(&req.params);
            }
            "textDocument/didClose" => {
                server.handle_did_close(&req.params);
            }
            "textDocument/completion" => {
                let _t0 = Instant::now();
                let result = server.handle_completion(&req.params);
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(result),
                        error: None,
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
            "textDocument/definition" => {
                let result = server.handle_definition(&req.params);
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(result),
                        error: None,
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
            "textDocument/hover" => {
                let result = server.handle_hover(&req.params);
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(result),
                        error: None,
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
            _ => {
                if let Some(id) = req.id {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32601,
                            message: format!("Method '{}' not found", req.method),
                            data: None,
                        }),
                    };
                    send_response(&mut writer, &resp)?;
                }
            }
        }
    }
}

fn send_response<W: Write>(writer: &mut W, resp: &JsonRpcResponse) -> Result<()> {
    let payload = serde_json::to_string(resp)?;
    write!(writer, "Content-Length: {}\r\n\r\n{}", payload.len(), payload)?;
    writer.flush()?;
    Ok(())
}
