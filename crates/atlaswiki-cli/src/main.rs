mod serve;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use ignore::WalkBuilder;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use atlaswiki_core::diagnostics::DiagnosticsEngine;
use atlaswiki_core::graph::KnowledgeGraph;
use atlaswiki_core::storage::StorageEngine;
use atlaswiki_core::watcher::{VaultWatcher, VaultWatcherConfig};
use atlaswiki_parser::MarkdownParser;
use serve::WebServer;

#[derive(Parser)]
#[command(
    name = "atlaswiki",
    version,
    about = "High-performance Markdown wiki indexer & semantic search engine",
    long_about = "AtlasWiki indexes local Markdown vaults (Obsidian, Foam, Logseq), parses extended graph syntax (wikilinks, transclusions, block refs, tags), and provides hybrid semantic search, bidirectional backlinks, graph analysis, diagnostics, and an interactive D3.js web visualizer."
)]
struct Cli {
    #[arg(short = 'C', long, global = true, help = "Path to markdown vault root (default: current directory)")]
    vault: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Recursively index Markdown files into high-speed SQLite database")]
    Index {
        #[arg(help = "Path to vault directory (default: current directory)")]
        path: Option<PathBuf>,
        #[arg(long, help = "Force full re-indexing of all files")]
        full: bool,
    },

    #[command(about = "Run hybrid full-text and semantic search")]
    Search {
        #[arg(help = "Search query string")]
        query: String,
        #[arg(short = 'n', long, default_value = "10", help = "Maximum results to return")]
        limit: usize,
        #[arg(long, help = "Output results in JSON format")]
        json: bool,
        #[arg(long, help = "Filter results by tag (e.g. #ai/ml)")]
        tag: Option<String>,
    },

    #[command(about = "Display note details, frontmatter, sections, outlinks, and backlinks")]
    Note {
        #[arg(help = "Note title or relative file path")]
        title: String,
        #[arg(long, help = "Output in JSON format")]
        json: bool,
    },

    #[command(about = "List incoming backlinks referencing a note")]
    Backlinks {
        #[arg(help = "Target note title")]
        title: String,
        #[arg(long, help = "Output in JSON format")]
        json: bool,
    },

    #[command(about = "Query knowledge graph metrics, shortest paths, and structural diagnostics")]
    Graph {
        #[command(subcommand)]
        command: GraphCommands,
        #[arg(long, global = true, help = "Output in JSON format")]
        json: bool,
    },

    #[command(about = "Check for broken links, missing headings, dead block references, and typos")]
    Check {
        #[arg(help = "Path to vault directory (default: current directory)")]
        path: Option<PathBuf>,
        #[arg(long, help = "Treat warnings as errors and exit with non-zero code")]
        strict: bool,
        #[arg(long, help = "Output diagnostics in JSON format")]
        json: bool,
    },

    #[command(about = "Launch embedded web server and interactive D3.js knowledge graph visualizer")]
    Serve {
        #[arg(help = "Path to vault directory (default: current directory)")]
        path: Option<PathBuf>,
        #[arg(short, long, default_value = "8888", help = "HTTP port to bind")]
        port: u16,
        #[arg(long, help = "Automatically open browser")]
        open: bool,
    },

    #[command(about = "Watch vault for filesystem changes and update index incrementally")]
    Watch {
        #[arg(help = "Path to vault directory (default: current directory)")]
        path: Option<PathBuf>,
    },

    #[command(about = "Launch Model Context Protocol (MCP) JSON-RPC stdio server")]
    Mcp {
        #[arg(help = "Path to vault directory (default: current directory)")]
        path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum GraphCommands {
    #[command(about = "Show general knowledge graph metrics (nodes, links, PageRank)")]
    Stats,
    #[command(about = "Find orphan notes (notes with no incoming or outgoing connections)")]
    Orphans,
    #[command(about = "Find wanted / dangling pages (referenced notes that do not exist yet)")]
    Wanted,
    #[command(about = "Find shortest conceptual path between Note A and Note B")]
    Path {
        #[arg(help = "Source note title")]
        from: String,
        #[arg(help = "Destination note title")]
        to: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let default_vault = cli.vault.unwrap_or_else(|| PathBuf::from("."));

    match cli.command {
        Commands::Index { path, full } => {
            let vault_root = path.unwrap_or(default_vault);
            cmd_index(&vault_root, full)?;
        }
        Commands::Search { query, limit, json, tag } => {
            let vault_root = default_vault;
            cmd_search(&vault_root, &query, limit, json, tag)?;
        }
        Commands::Note { title, json } => {
            let vault_root = default_vault;
            cmd_note(&vault_root, &title, json)?;
        }
        Commands::Backlinks { title, json } => {
            let vault_root = default_vault;
            cmd_backlinks(&vault_root, &title, json)?;
        }
        Commands::Graph { command, json } => {
            let vault_root = default_vault;
            cmd_graph(&vault_root, command, json)?;
        }
        Commands::Check { path, strict, json } => {
            let vault_root = path.unwrap_or(default_vault);
            cmd_check(&vault_root, strict, json)?;
        }
        Commands::Serve { path, port, open } => {
            let vault_root = path.unwrap_or(default_vault);
            cmd_serve(&vault_root, port, open)?;
        }
        Commands::Watch { path } => {
            let vault_root = path.unwrap_or(default_vault);
            cmd_watch(&vault_root)?;
        }
        Commands::Mcp { path } => {
            let vault_root = path.unwrap_or(default_vault);
            cmd_mcp(&vault_root)?;
        }
    }

    Ok(())
}

fn open_storage(vault_root: &Path) -> Result<StorageEngine> {
    let atlas_dir = vault_root.join(".atlaswiki");
    let db_path = atlas_dir.join("index.db");
    StorageEngine::open(&db_path).with_context(|| format!("Failed to open index database at {:?}", db_path))
}

fn cmd_index(vault_root: &Path, full: bool) -> Result<()> {
    let start_time = Instant::now();
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    println!(
        "{} Indexing vault at {}",
        "⚙".cyan().bold(),
        canonical_vault.display().to_string().bold()
    );

    let storage = open_storage(&canonical_vault)?;
    let parser = MarkdownParser::new();

    // Collect all markdown files
    let mut files = Vec::new();
    let walker = WalkBuilder::new(&canonical_vault)
        .hidden(true) // do not traverse into hidden folders
        .git_ignore(true)
        .build();

    for entry in walker.filter_map(Result::ok) {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|ext| ext == "md") {
            let rel = match p.strip_prefix(&canonical_vault) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => continue,
            };

            // Skip internal files
            if rel.starts_with(".atlaswiki") || rel.starts_with(".git") || rel.starts_with(".obsidian") {
                continue;
            }

            files.push((p.to_path_buf(), rel));
        }
    }

    let total_discovered = files.len();
    println!("  Discovered {} markdown notes.", total_discovered.to_string().cyan());

    // Filter files needing update
    let mut to_process = Vec::new();
    for (abs, rel) in files {
        let metadata = match fs::metadata(&abs) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let size = metadata.len();

        if !full {
            if let Ok(Some((stored_hash, stored_mtime, stored_size))) = storage.get_manifest(&rel) {
                if stored_mtime == mtime && stored_size == size {
                    continue; // Unmodified fast-path!
                }

                // Verify hash if mtime differed
                if let Ok(content) = fs::read_to_string(&abs) {
                    let mut hasher = Sha256::new();
                    hasher.update(content.as_bytes());
                    let current_hash = format!("{:x}", hasher.finalize());
                    if current_hash == stored_hash {
                        continue;
                    }
                }
            }
        }

        to_process.push((abs, rel, mtime, size));
    }

    let files_to_index = to_process.len();
    if files_to_index == 0 {
        println!("{} All notes are up to date. Zero changes detected.", "✓".green().bold());
        return Ok(());
    }

    println!("  Parsing & indexing {} modified notes...", files_to_index.to_string().yellow());

    // Parse in parallel with Rayon
    let parsed_docs: Vec<_> = to_process
        .into_par_iter()
        .filter_map(|(abs, rel, mtime, size)| {
            let content = fs::read_to_string(&abs).ok()?;
            let doc = parser.parse_file(Path::new(&rel), &content).ok()?;
            Some((doc, rel, mtime, size))
        })
        .collect();

    // Atomic SQLite insertion
    for (doc, rel, mtime, size) in &parsed_docs {
        let doc_id = format!("doc_{}", &doc.content_hash[..16]);
        storage.sync_document(doc, &doc_id, rel, *mtime, *size, None)?;
    }

    let stats = storage.get_stats()?;
    let elapsed = start_time.elapsed();

    println!(
        "\n{} Indexed {} notes in {:.2?}",
        "✓".green().bold(),
        files_to_index.to_string().bold(),
        elapsed
    );
    println!("  Total Documents: {}", stats.total_documents.to_string().cyan());
    println!("  Total Sections:  {}", stats.total_sections.to_string().cyan());
    println!("  Total Links:     {}", stats.total_links.to_string().cyan());
    println!("  Total Chunks:    {}", stats.total_chunks.to_string().cyan());
    println!("  Total Words:     {}", stats.total_words.to_string().cyan());

    Ok(())
}

fn cmd_search(
    vault_root: &Path,
    query: &str,
    limit: usize,
    json: bool,
    tag_filter: Option<String>,
) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = open_storage(&canonical_vault)?;

    let results = storage.search_fts(query, limit * 2)?;
    let filtered_results: Vec<_> = results
        .into_iter()
        .filter(|(_, title, _, _, _)| {
            if let Some(ref tag) = tag_filter {
                let clean_tag = tag.trim_start_matches('#');
                title.to_lowercase().contains(&clean_tag.to_lowercase())
            } else {
                true
            }
        })
        .take(limit)
        .collect();

    if json {
        let hits: Vec<_> = filtered_results
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
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }

    if filtered_results.is_empty() {
        println!("{} No results found for query: '{}'", "ℹ".yellow(), query);
        return Ok(());
    }

    println!(
        "\n{} Found {} results for '{}':\n",
        "✓".green().bold(),
        filtered_results.len().to_string().bold(),
        query.cyan()
    );

    for (i, (_chunk_id, title, breadcrumbs, snippet, _score)) in filtered_results.into_iter().enumerate() {
        println!(
            "{}. {} {}",
            (i + 1).to_string().bold(),
            title.cyan().bold(),
            format!("({})", breadcrumbs).dimmed()
        );
        let clean_snippet = snippet
            .replace("<b>", "\x1b[1;33m")
            .replace("</b>", "\x1b[0m");
        println!("   {}\n", clean_snippet.trim());
    }

    Ok(())
}

fn cmd_note(vault_root: &Path, title: &str, json: bool) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = open_storage(&canonical_vault)?;

    let doc = storage
        .get_document(title)?
        .ok_or_else(|| anyhow::anyhow!("Note '{}' not found in vault index.", title))?;

    let sections = storage.get_sections_for_doc(&doc.doc_id)?;
    let outlinks = storage.get_outlinks_for_doc(&doc.doc_id)?;
    let backlinks = storage.get_backlinks(&doc.title)?;

    if json {
        let payload = serde_json::json!({
            "document": doc,
            "sections": sections,
            "outlinks": outlinks,
            "backlinks": backlinks
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("\n{} {}", "Note:".bold(), doc.title.cyan().bold());
    println!("  Path:        {}", doc.path.dimmed());
    println!("  Word Count:  {}", doc.word_count);

    if !sections.is_empty() {
        println!("\n{}", "Sections:".bold());
        for sec in &sections {
            let indent = "  ".repeat(sec.level);
            println!("{}- {} (lines {}-{})", indent, sec.heading.bold(), sec.line_start, sec.line_end);
        }
    }

    if !outlinks.is_empty() {
        println!("\n{}", "Outgoing Links:".bold());
        for link in &outlinks {
            let heading = link.target_heading.as_ref().map(|h| format!("#{h}")).unwrap_or_default();
            println!("  → [[{}{}]] (line {})", link.target_note.cyan(), heading, link.line_number);
        }
    }

    if !backlinks.is_empty() {
        println!("\n{}", "Incoming Backlinks:".bold());
        for bl in &backlinks {
            println!("  ← [[{}]] (line {})", bl.source_title.yellow(), bl.line_number);
            if let Some(ref snip) = bl.snippet {
                println!("    \"{}\"", snip.italic().dimmed());
            }
        }
    }

    println!();
    Ok(())
}

fn cmd_backlinks(vault_root: &Path, title: &str, json: bool) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = open_storage(&canonical_vault)?;
    let backlinks = storage.get_backlinks(title)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&backlinks)?);
        return Ok(());
    }

    if backlinks.is_empty() {
        println!("{} No backlinks found pointing to [[{}]]", "ℹ".yellow(), title);
        return Ok(());
    }

    println!(
        "\n{} {} backlinks found pointing to [[{}]]:\n",
        "✓".green().bold(),
        backlinks.len().to_string().bold(),
        title.cyan()
    );

    for bl in backlinks {
        println!(
            "  • [[{}]] {} (line {})",
            bl.source_title.cyan().bold(),
            bl.source_path.dimmed(),
            bl.line_number
        );
        if let Some(snip) = bl.snippet {
            println!("    \"{}\"", snip.trim().italic().dimmed());
        }
    }

    println!();
    Ok(())
}

fn cmd_graph(vault_root: &Path, command: GraphCommands, json: bool) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = open_storage(&canonical_vault)?;

    // Build knowledge graph from indexed documents
    let parser = MarkdownParser::new();
    let paths = storage.get_all_document_paths()?;
    let mut docs = Vec::new();

    for p in paths {
        let full_p = canonical_vault.join(&p);
        if let Ok(content) = fs::read_to_string(&full_p) {
            if let Ok(doc) = parser.parse_file(&full_p, &content) {
                docs.push(doc);
            }
        }
    }

    let mut graph = KnowledgeGraph::from_documents(&docs);
    graph.compute_pagerank(0.85, 50);

    match command {
        GraphCommands::Stats => {
            let stats = storage.get_stats()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("\n{}", "Vault Knowledge Graph Metrics:".bold());
                println!("  Documents:   {}", stats.total_documents.to_string().cyan());
                println!("  Sections:    {}", stats.total_sections.to_string().cyan());
                println!("  Connections: {}", stats.total_links.to_string().cyan());
                println!("  Tags:        {}", stats.total_tags.to_string().cyan());
                println!("  Chunks:      {}", stats.total_chunks.to_string().cyan());
                println!("  Total Words: {}", stats.total_words.to_string().cyan());
                println!();
            }
        }
        GraphCommands::Orphans => {
            let orphans = graph.get_orphans();
            if json {
                println!("{}", serde_json::to_string_pretty(&orphans)?);
            } else {
                println!("\n{} Isolated Orphan Notes (0 in, 0 out):\n", "ℹ".yellow().bold());
                if orphans.is_empty() {
                    println!("  No orphan notes detected in vault.");
                } else {
                    for o in orphans {
                        println!("  • [[{}]]", o.title.yellow());
                    }
                }
                println!();
            }
        }
        GraphCommands::Wanted => {
            let wanted = graph.get_wanted_pages();
            if json {
                println!("{}", serde_json::to_string_pretty(&wanted)?);
            } else {
                println!("\n{} Wanted / Dangling Pages (Unwritten Notes):\n", "⚡".yellow().bold());
                if wanted.is_empty() {
                    println!("  All wikilinks resolve to existing notes.");
                } else {
                    for (target, count) in wanted {
                        println!("  • [[{}]] (referenced {} times)", target.red(), count.to_string().bold());
                    }
                }
                println!();
            }
        }
        GraphCommands::Path { from, to } => {
            let path = graph.shortest_path(&from, &to);
            if json {
                println!("{}", serde_json::to_string_pretty(&path)?);
            } else {
                println!("\n{} Shortest Path from [[{}]] to [[{}]]:\n", "⚲".cyan().bold(), from.cyan(), to.cyan());
                if let Some(nodes) = path {
                    let formatted = nodes
                        .iter()
                        .map(|n| format!("[[{}]]", n.cyan().bold()))
                        .collect::<Vec<_>>()
                        .join("  →  ");
                    println!("  {}", formatted);
                } else {
                    println!("  No connected path found between [[{}]] and [[{}]].", from, to);
                }
                println!();
            }
        }
    }

    Ok(())
}

fn cmd_check(vault_root: &Path, strict: bool, json: bool) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let parser = MarkdownParser::new();
    let mut diagnostics = DiagnosticsEngine::new();

    // Read all markdown files
    let mut docs = Vec::new();
    let walker = WalkBuilder::new(&canonical_vault)
        .hidden(true)
        .git_ignore(true)
        .build();

    for entry in walker.filter_map(Result::ok) {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|ext| ext == "md") {
            let rel = match p.strip_prefix(&canonical_vault) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => continue,
            };

            if rel.starts_with(".atlaswiki") || rel.starts_with(".git") || rel.starts_with(".obsidian") {
                continue;
            }

            if let Ok(content) = fs::read_to_string(p) {
                if let Ok(doc) = parser.parse_file(Path::new(&rel), &content) {
                    diagnostics.index_document(&doc);
                    docs.push(doc);
                }
            }
        }
    }

    let report = diagnostics.run(&docs, strict);

    if json {
        println!("{}", report.to_json()?);
        if strict && report.error_count() > 0 {
            std::process::exit(1);
        }
        return Ok(());
    }

    println!(
        "\n{} Scanned {} notes, checked {} links.\n",
        "✓".green().bold(),
        report.total_files_scanned.to_string().bold(),
        report.total_links_checked.to_string().bold()
    );

    if report.diagnostics.is_empty() {
        println!("{} Vault is completely healthy! Zero dead links or anchors found.\n", "✓".green().bold());
        return Ok(());
    }

    for diag in &report.diagnostics {
        let code_badge = match diag.severity {
            atlaswiki_core::diagnostics::DiagnosticSeverity::Error => format!("error[{}]", diag.code.code_str()).red().bold(),
            atlaswiki_core::diagnostics::DiagnosticSeverity::Warning => format!("warning[{}]", diag.code.code_str()).yellow().bold(),
            atlaswiki_core::diagnostics::DiagnosticSeverity::Info => format!("info[{}]", diag.code.code_str()).blue().bold(),
        };

        println!(
            "{}: {}:{} - {}",
            code_badge,
            diag.location.file_path.display().to_string().bold(),
            diag.location.line_number,
            diag.message
        );
        if !diag.location.source_line.is_empty() {
            println!("   | {}", diag.location.source_line.dimmed());
        }
        if let Some(ref sug) = diag.suggestion {
            println!("   = {} {}\n", "help: did you mean:".cyan().bold(), sug.green().bold());
        } else {
            println!();
        }
    }

    let err_count = report.error_count();
    let warn_count = report.warning_count();

    println!(
        "Result: {} errors, {} warnings found.\n",
        err_count.to_string().red().bold(),
        warn_count.to_string().yellow().bold()
    );

    if strict && err_count > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn cmd_serve(vault_root: &Path, port: u16, open: bool) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = Arc::new(open_storage(&canonical_vault)?);
    let server = WebServer::new(&canonical_vault, storage, port);

    if open {
        let url = format!("http://localhost:{}", port);
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&url).spawn();
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("explorer").arg(&url).spawn();
    }

    server.run()
}

fn cmd_watch(vault_root: &Path) -> Result<()> {
    let canonical_vault = vault_root.canonicalize().context("Vault path does not exist")?;
    let storage = open_storage(&canonical_vault)?;

    let config = VaultWatcherConfig {
        vault_root: canonical_vault.clone(),
        ..Default::default()
    };

    println!(
        "{} Watching vault at {} for changes...",
        "👁".cyan().bold(),
        canonical_vault.display().to_string().bold()
    );
    println!("{}", "Press Ctrl+C to stop.\n".dimmed());

    let (_watcher, rx) = VaultWatcher::start(config, storage)?;

    for event in rx {
        match event {
            atlaswiki_core::watcher::VaultSyncEvent::NoteIndexed { path, word_count, chunks_count } => {
                println!(
                    "{} Indexed note: {} ({} words, {} chunks)",
                    "✓".green(),
                    path.display().to_string().bold(),
                    word_count,
                    chunks_count
                );
            }
            atlaswiki_core::watcher::VaultSyncEvent::NoteDeleted { path } => {
                println!("{} Deleted note: {}", "✗".red(), path.display().to_string().bold());
            }
            atlaswiki_core::watcher::VaultSyncEvent::NoteRenamed(r) => {
                println!(
                    "{} Renamed note: {} → {}",
                    "→".yellow(),
                    r.old_path.display(),
                    r.new_path.display().to_string().bold()
                );
            }
            atlaswiki_core::watcher::VaultSyncEvent::SyncError { path, error } => {
                eprintln!("{} Error syncing {}: {}", "⚠".red(), path.display(), error);
            }
        }
    }

    Ok(())
}

fn cmd_mcp(vault_root: &Path) -> Result<()> {
    // Launch atlaswiki-mcp subprocess
    let status = std::process::Command::new("atlaswiki-mcp")
        .arg("-C")
        .arg(vault_root)
        .status()?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
