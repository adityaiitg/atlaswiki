# AtlasWiki 🪐

[![Rust](https://img.shields.io/badge/rust-2021%20edition-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AtlasWiki** is an ultra-fast, local-first personal knowledge base (PKB) indexer, knowledge graph engine, and hybrid semantic search system designed for Markdown vaults (compatible with Obsidian, Logseq, Foam, and Dendron).

Engineered in **100% pure Rust**, AtlasWiki requires **zero C/C++ or Python runtime dependencies**, avoids ONNX shared library bindings, operates **strictly non-destructively** over user notes, and stores all indices in high-speed SQLite WAL databases.

---

## Key Features

- 🌳 **Extended Markdown AST Parser (`atlaswiki-parser`)**:
  - Full support for `[[Wikilinks]]`, `[[Note|Alias]]`, `[[Note#Heading]]`, and `[[Note#^block-id]]`.
  - Transclusions & embeds `![[Note]]` and `![[Note#Heading]]`.
  - Hierarchical nested tags (`#category/subcategory`), ignoring `#` in hex colors, numbers, or headings.
  - End-of-line block identifiers `^block-id` for granular referencing.
  - Code block and inline code span isolation (never triggers false positive links or tags).
  - Robust YAML frontmatter parsing (`title`, `aliases`, `tags`, arbitrary metadata).
  - Semantic section chunking that preserves structural ancestry breadcrumbs (`Note > H1 > H2`).

- ⚡ **SQLite WAL Storage Engine (`atlaswiki-core`)**:
  - Full relational persistence with Foreign Keys and ON DELETE CASCADE.
  - SQLite FTS5 full-text search with Unicode61 tokenizer and Porter stemming.
  - Automated insert/update/delete SQLite triggers synchronizing FTS virtual tables.
  - Incremental sync engine tracking SHA-256 hashes and mtimes for sub-second index refresh.

- 🕸️ **Bidirectional Knowledge Graph & Graph Theory**:
  - In-memory `petgraph` directional graph (`DiGraph<NoteNode, LinkEdge>`).
  - Dangling-Safe PageRank power iteration to surface Map-of-Content (MOC) cornerstone notes.
  - Bidirectional BFS shortest conceptual pathfinder between any two notes (e.g., `Quantum -> Information Theory -> Cryptography`).
  - Structural diagnostics: automated orphan detection (0 in, 0 out) and wanted unwritten pages.

- 🔍 **Pure-Rust Semantic Retrieval (Model2Vec Engine)**:
  - Static embedding lookup via Hugging Face `safetensors` and `tokenizers` (pure Rust `fancy-regex` backend).
  - Zero C++ / zero ONNX shared library runtime.
  - Multi-signal Reciprocal Rank Fusion (RRF with $k=60$) combining BM25 and vector cosine similarity.
  - PKB-specific reranking boosts: note title exact match ($2.5\times$), heading match ($2.0\times$), tag match ($1.5\times$), and PageRank centrality (+20%).

- 🌐 **Interactive D3.js Web Graph Visualizer (`atlaswiki serve`)**:
  - Embedded zero-dependency HTTP server (`tiny_http`).
  - Dark Obsidian/GitHub theme canvas/SVG force-directed graph.
  - Tag cluster color palette, PageRank node sizing, real-time search filter.
  - Slide-in inspection drawer showing note preview, frontmatter, and incoming backlinks.
  - Focus mode isolating 1-hop and 2-hop neighborhood subgraphs.

- 🩺 **Dead Link & Typo Diagnostics Linter (`atlaswiki check`)**:
  - Detects dead wikilinks, broken heading anchors, and missing block references.
  - Embedded Damerau-Levenshtein typo correction engine suggesting `Did you mean [[Machine Learning]]?`.
  - Compiler-style terminal formatting with exact line and column pointers.

- 🤖 **Model Context Protocol (MCP) Server (`atlaswiki-mcp`)**:
  - Conforms to MCP stdio JSON-RPC 2.0 specification (`2024-11-05`).
  - Exposes 6 AI tools for Cursor, Claude, Codex, and Gemini (`atlaswiki_search`, `atlaswiki_get_note`, `atlaswiki_backlinks`, `atlaswiki_graph_neighbors`, `atlaswiki_unresolved_links`, `atlaswiki_stats`).

---

## Architecture Overview

```
atlaswiki/
├── crates/
│   ├── atlaswiki-parser/   # Extended Markdown lexer, AST, frontmatter, semantic chunker
│   ├── atlaswiki-core/     # SQLite WAL storage, knowledge graph, Model2Vec, watcher, diagnostics
│   ├── atlaswiki-cli/      # CLI binary with clap v4, embedded tiny_http & D3.js visualizer
│   └── atlaswiki-mcp/      # Model Context Protocol stdio JSON-RPC server
└── tests/
    └── fixtures/sample_vault/  # Comprehensive reference test vault
```

---

## Installation & Building

Prerequisites: Rust 1.80+ (stable).

```bash
# Clone the repository
git clone https://github.com/adityaiitg/atlaswiki.git
cd atlaswiki

# Build all workspace crates in release mode
cargo build --release

# Run all test suites
cargo test --workspace

# Install atlaswiki and atlaswiki-mcp to ~/.cargo/bin
cargo install --path crates/atlaswiki-cli
cargo install --path crates/atlaswiki-mcp
```

---

## CLI Usage

### 1. Indexing a Vault
```bash
# Index current directory or specified vault
atlaswiki index /path/to/vault

# Force a full re-indexing of all files
atlaswiki index /path/to/vault --full
```

### 2. Hybrid Search
```bash
# Search vault notes and chunks
atlaswiki search "transformer self-attention" -C /path/to/vault

# Output results as JSON (for piping or editor integration)
atlaswiki search "neural networks" --json -C /path/to/vault

# Filter search results by tag
atlaswiki search "gradient descent" --tag "#ai/ml" -C /path/to/vault
```

### 3. Note & Backlinks Inspection
```bash
# Inspect note metadata, sections hierarchy, outlinks, and backlinks
atlaswiki note "Machine Learning" -C /path/to/vault

# List all incoming backlinks referencing a note
atlaswiki backlinks "Machine Learning" -C /path/to/vault
```

### 4. Knowledge Graph Analysis
```bash
# General vault metrics (nodes, links, chunks, words)
atlaswiki graph stats -C /path/to/vault

# List orphan notes (notes with zero incoming and outgoing links)
atlaswiki graph orphans -C /path/to/vault

# List wanted / dangling pages (referenced notes that do not exist yet)
atlaswiki graph wanted -C /path/to/vault

# Find the shortest conceptual path between two notes
atlaswiki graph path "AtlasWiki Index" "Neural Networks" -C /path/to/vault
```

### 5. Link Diagnostics Linter
```bash
# Scan vault for dead wikilinks, broken headings, and missing blocks
atlaswiki check /path/to/vault

# Strict mode (fails CI with non-zero exit code if warnings exist)
atlaswiki check /path/to/vault --strict
```

### 6. Interactive Web Visualizer
```bash
# Launch embedded HTTP web server on default port 8888
atlaswiki serve /path/to/vault

# Specify custom port and open browser automatically
atlaswiki serve /path/to/vault --port 9000 --open
```

### 7. Real-Time File Watcher
```bash
# Continuously watch vault and update SQLite index incrementally
atlaswiki watch /path/to/vault
```

### 8. AI Agent Model Context Protocol (MCP)
```bash
# Launch MCP stdio server
atlaswiki mcp /path/to/vault
```

To configure in Claude Desktop or Cursor (`~/.config/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "atlaswiki": {
      "command": "atlaswiki-mcp",
      "args": ["-C", "/absolute/path/to/your/markdown/vault"]
    }
  }
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
