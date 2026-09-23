//! Model Context Protocol (MCP) stdio server for AtlasWiki.
//!
//! Exposes AtlasWiki search, note reading, backlinks, graph neighborhood,
//! unresolved links, and vault stats tools to LLMs (Claude, Cursor, Codex, Gemini).

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use atlaswiki_core::graph::KnowledgeGraph;
use atlaswiki_core::storage::StorageEngine;
use atlaswiki_parser::MarkdownParser;

#[derive(Parser, Debug)]
#[command(
    name = "atlaswiki-mcp",
    version,
    about = "Model Context Protocol (MCP) server for AtlasWiki"
)]
struct Args {
    #[arg(short = 'C', long, help = "Path to vault directory (default: current directory)")]
    vault: Option<PathBuf>,
}

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

fn main() -> Result<()> {
    let args = Args::parse();
    let vault_root = args.vault.unwrap_or_else(|| PathBuf::from("."));
    let canonical_vault = vault_root
        .canonicalize()
        .context("Vault path does not exist")?;

    eprintln!(
        "[atlaswiki-mcp] Starting MCP server for vault: {}",
        canonical_vault.display()
    );

    let db_path = canonical_vault.join(".atlaswiki").join("index.db");
    let storage = Arc::new(
        StorageEngine::open(&db_path)
            .with_context(|| format!("Failed to open index at {:?}", db_path))?,
    );

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[atlaswiki-mcp] Stdin error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[atlaswiki-mcp] JSON parse error: {e}");
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
                let _ = writeln!(handle, "{}", serde_json::to_string(&err_resp)?);
                let _ = handle.flush();
                continue;
            }
        };

        // Process request
        let response = handle_request(&req, &canonical_vault, &storage);

        // Send response if not a notification
        if let Some(resp) = response {
            let out_str = serde_json::to_string(&resp)?;
            writeln!(handle, "{out_str}")?;
            handle.flush()?;
        }
    }

    eprintln!("[atlaswiki-mcp] Server shut down.");
    Ok(())
}

fn handle_request(
    req: &JsonRpcRequest,
    vault_root: &Path,
    storage: &Arc<StorageEngine>,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone().unwrap_or(Value::Null);

    match req.method.as_str() {
        "initialize" => {
            let result = json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "atlaswiki-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            });
            Some(success_response(id, result))
        }
        "notifications/initialized" => None,
        "ping" => Some(success_response(id, json!({}))),
        "tools/list" => {
            let tools = list_tools();
            Some(success_response(id, json!({ "tools": tools })))
        }
        "tools/call" => {
            let tool_name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = req.params.get("arguments").cloned().unwrap_or(json!({}));

            match call_tool(tool_name, &arguments, vault_root, storage) {
                Ok(content) => Some(success_response(
                    id,
                    json!({
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }),
                )),
                Err(e) => Some(error_response(id, -32000, e.to_string())),
            }
        }
        unknown => {
            eprintln!("[atlaswiki-mcp] Unknown method: {unknown}");
            Some(error_response(
                id,
                -32601,
                format!("Method not found: {unknown}"),
            ))
        }
    }
}

fn success_response(id: Value, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Value, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message,
            data: None,
        }),
    }
}

fn list_tools() -> Vec<Value> {
    vec![
        json!({
            "name": "atlaswiki_search",
            "description": "Perform full-text BM25 and semantic hybrid search across notes and chunks in the AtlasWiki knowledge base.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, phrase, or keyword."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default: 10)."
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "atlaswiki_get_note",
            "description": "Retrieve full markdown content, frontmatter metadata, sections hierarchy, outgoing links, and incoming backlinks for a given note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title or relative path of the note to retrieve."
                    }
                },
                "required": ["title"]
            }
        }),
        json!({
            "name": "atlaswiki_backlinks",
            "description": "Get all incoming backlinks referencing a note, including referencing files, line numbers, and context snippets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the note to inspect incoming references for."
                    }
                },
                "required": ["title"]
            }
        }),
        json!({
            "name": "atlaswiki_graph_neighbors",
            "description": "Retrieve the local subgraph (1-hop or 2-hop connected neighbors and wikilink connections) around a given note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The center note title."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Hop distance (1 or 2, default: 1)."
                    }
                },
                "required": ["title"]
            }
        }),
        json!({
            "name": "atlaswiki_unresolved_links",
            "description": "List all unresolved or dangling wikilinks in the vault sorted by reference frequency (wanted unwritten notes).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "atlaswiki_stats",
            "description": "Get comprehensive metrics for the AtlasWiki knowledge vault (notes count, sections, links, chunks, words).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "atlaswiki_graph_rag",
            "description": "Perform Graph RAG multi-hop connective path and Steiner-tree context extraction between concept notes for LLM reasoning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of concept note titles to connect."
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum path length in hops (default: 3)."
                    }
                },
                "required": ["seeds"]
            }
        }),
        json!({
            "name": "atlaswiki_generate_moc",
            "description": "Generate or update an automated Map of Content (MOC) index based on tag taxonomy, PageRank centrality, and backlinks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic or tag to generate MOC for. If omitted, generates living vault index."
                    }
                }
            }
        }),
    ]
}

fn call_tool(
    name: &str,
    args: &Value,
    vault_root: &Path,
    storage: &Arc<StorageEngine>,
) -> Result<String> {
    match name {
        "atlaswiki_search" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .context("Missing 'query' parameter")?;
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;

            let hits = storage.search_fts(query, limit)?;
            let results: Vec<Value> = hits
                .into_iter()
                .map(|(chunk_id, title, breadcrumbs, snippet, score)| {
                    json!({
                        "chunk_id": chunk_id,
                        "title": title,
                        "breadcrumbs": breadcrumbs,
                        "snippet": snippet,
                        "score": score
                    })
                })
                .collect();

            Ok(serde_json::to_string_pretty(&results)?)
        }
        "atlaswiki_get_note" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .context("Missing 'title' parameter")?;

            let doc = storage
                .get_document(title)?
                .ok_or_else(|| anyhow::anyhow!("Note '{}' not found in vault index", title))?;

            let vault = atlaswiki_core::security::VaultRoot::new(&vault_root)?;
            let content = vault.safe_read_file(&doc.path).unwrap_or_default();
            let sections = storage.get_sections_for_doc(&doc.doc_id)?;
            let outlinks = storage.get_outlinks_for_doc(&doc.doc_id)?;
            let backlinks = storage.get_backlinks(&doc.title)?;

            let payload = json!({
                "document": doc,
                "sections": sections,
                "outlinks": outlinks,
                "backlinks": backlinks,
                "content": content
            });

            Ok(serde_json::to_string_pretty(&payload)?)
        }
        "atlaswiki_backlinks" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .context("Missing 'title' parameter")?;

            let backlinks = storage.get_backlinks(title)?;
            Ok(serde_json::to_string_pretty(&backlinks)?)
        }
        "atlaswiki_graph_neighbors" => {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .context("Missing 'title' parameter")?;

            let parser = MarkdownParser::new();
            let paths = storage.get_all_document_paths()?;
            let mut docs = Vec::new();

            for p in paths {
                let full_p = vault_root.join(&p);
                if let Ok(c) = fs::read_to_string(&full_p) {
                    if let Ok(doc) = parser.parse_file(&full_p, &c) {
                        docs.push(doc);
                    }
                }
            }

            let graph = KnowledgeGraph::from_documents(&docs);
            let backlinks = graph.get_backlinks(title);
            let outlinks = storage
                .get_document(title)?
                .map(|d| storage.get_outlinks_for_doc(&d.doc_id).unwrap_or_default())
                .unwrap_or_default();

            let payload = json!({
                "center_note": title,
                "incoming_links": backlinks.into_iter().map(|(n, _e)| n.title).collect::<Vec<_>>(),
                "outgoing_links": outlinks.into_iter().map(|l| l.target_note).collect::<Vec<_>>()
            });

            Ok(serde_json::to_string_pretty(&payload)?)
        }
        "atlaswiki_unresolved_links" => {
            let unresolved = storage.get_unresolved_links()?;
            Ok(serde_json::to_string_pretty(&unresolved)?)
        }
        "atlaswiki_stats" => {
            let stats = storage.get_stats()?;
            Ok(serde_json::to_string_pretty(&stats)?)
        }
        "atlaswiki_graph_rag" => {
            let seeds_val = args
                .get("seeds")
                .and_then(|v| v.as_array())
                .context("Missing or invalid 'seeds' array")?;
            let seeds: Vec<&str> = seeds_val.iter().filter_map(|v| v.as_str()).collect();
            let max_hops = args
                .get("max_hops")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;

            let engine = atlaswiki_core::graph_rag::GraphRagEngine::from_storage(storage)?;
            let result = engine.extract_context(&seeds, max_hops, 15, true);
            Ok(result.markdown_context)
        }
        "atlaswiki_generate_moc" => {
            let topic = args.get("topic").and_then(|v| v.as_str());
            let synthesizer = atlaswiki_core::synthesis::MocSynthesizer::new(vault_root);
            if let Some(t) = topic {
                let report = synthesizer.generate_moc(Some(t), false)?;
                Ok(serde_json::to_string_pretty(&report)?)
            } else {
                let report = synthesizer.generate_living_index(false)?;
                Ok(serde_json::to_string_pretty(&report)?)
            }
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}
