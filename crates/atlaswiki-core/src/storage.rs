//! SQLite WAL Storage Engine for AtlasWiki.
//!
//! Provides relational persistence for documents, sections, wikilinks, tags,
//! semantic chunks, FTS5 full-text index with triggers, and vector embeddings.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::{Context, Result};
use rayon::prelude::*;
use rusqlite::{params, Connection, OpenFlags, Transaction, TransactionBehavior};
use thiserror::Error;

use atlaswiki_parser::{LinkType, ParsedDocument};

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid embedding byte length: expected multiple of 4, got {0}")]
    InvalidEmbeddingLength(usize),
    #[error("Document not found: {0}")]
    DocumentNotFound(String),
}

/// Statistics about the knowledge vault index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VaultStats {
    pub total_documents: usize,
    pub total_sections: usize,
    pub total_links: usize,
    pub total_tags: usize,
    pub total_chunks: usize,
    pub total_embeddings: usize,
    pub total_words: usize,
}

/// A stored backlink with source note, line number, heading, and snippet.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BacklinkRecord {
    pub source_doc_id: String,
    pub source_path: String,
    pub source_title: String,
    pub line_number: usize,
    pub link_type: String,
    pub target_heading: Option<String>,
    pub target_block: Option<String>,
    pub alias: Option<String>,
    pub snippet: Option<String>,
}

/// An unresolved or dangling link reference.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnresolvedLinkRecord {
    pub target_note: String,
    pub reference_count: usize,
    pub source_notes: Vec<String>,
}

/// A stored document record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub path: String,
    pub title: String,
    pub frontmatter_json: String,
    pub word_count: usize,
    pub mtime: u64,
    pub hash: String,
}

/// A stored section record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SectionRecord {
    pub section_id: String,
    pub doc_id: String,
    pub heading: String,
    pub level: usize,
    pub parent_id: Option<String>,
    pub breadcrumbs: Vec<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
}

/// A stored semantic chunk record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkRecord {
    pub chunk_id: String,
    pub doc_id: String,
    pub section_id: String,
    pub title: String,
    pub breadcrumbs: Vec<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub content: String,
}

/// A stored outbound link record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinkRecord {
    pub source_doc_id: String,
    pub target_note: String,
    pub target_heading: Option<String>,
    pub target_block: Option<String>,
    pub alias: Option<String>,
    pub link_type: String,
    pub line_number: usize,
    pub context_snippet: Option<String>,
}

/// Thread-safe storage engine providing a dedicated writer and read connections.
#[derive(Clone)]
pub struct StorageEngine {
    db_path: PathBuf,
    writer: Arc<Mutex<Connection>>,
}

impl StorageEngine {
    /// Opens or creates the AtlasWiki SQLite database with optimal WAL configuration.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db_path = path.as_ref().to_path_buf();

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create DB directory: {:?}", parent))?;
        }

        let mut writer_conn = Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        Self::apply_pragmas(&mut writer_conn, true)?;
        Self::init_schema(&mut writer_conn)?;

        Ok(Self {
            db_path,
            writer: Arc::new(Mutex::new(writer_conn)),
        })
    }

    /// Opens an in-memory database configured for tests.
    pub fn open_in_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory()?;
        Self::apply_pragmas(&mut conn, true)?;
        Self::init_schema(&mut conn)?;

        Ok(Self {
            db_path: PathBuf::from(":memory:"),
            writer: Arc::new(Mutex::new(conn)),
        })
    }

    /// Spawns an isolated reader connection with read-optimized PRAGMAs.
    pub fn reader(&self) -> Result<Connection> {
        if self.db_path.to_str() == Some(":memory:") {
            // In-memory mode returns error or clone; readers shouldn't open separate memory DB
            return Err(anyhow::anyhow!("In-memory database cannot spawn separate file readers; use writer or clone connection"));
        }

        let mut reader_conn = Connection::open_with_flags(
            &self.db_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        Self::apply_pragmas(&mut reader_conn, false)?;
        Ok(reader_conn)
    }

    /// Executes a read closure using the writer lock (for in-memory compatibility).
    pub fn with_read_conn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R>,
    {
        if self.db_path.to_str() == Some(":memory:") {
            let guard = self.writer.lock().unwrap();
            f(&guard)
        } else {
            let reader = self.reader()?;
            f(&reader)
        }
    }

    /// Applies high-speed SQLite PRAGMAs.
    fn apply_pragmas(conn: &mut Connection, is_writer: bool) -> Result<()> {
        if is_writer {
            let _ = conn.query_row("PRAGMA journal_mode = WAL;", [], |r| r.get::<_, String>(0));
            let _ = conn.query_row("PRAGMA wal_autocheckpoint = 1000;", [], |r| r.get::<_, i64>(0));
            let _ = conn.execute("PRAGMA synchronous = NORMAL;", []);
        }

        let _ = conn.query_row("PRAGMA mmap_size = 268435456;", [], |r| r.get::<_, i64>(0));
        let _ = conn.execute("PRAGMA cache_size = -64000;", []);
        let _ = conn.execute("PRAGMA temp_store = MEMORY;", []);
        let _ = conn.execute("PRAGMA foreign_keys = ON;", []);
        let _ = conn.execute("PRAGMA recursive_triggers = ON;", []);

        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(())
    }

    /// Idempotently initializes database tables, virtual tables, triggers, and indices.
    pub fn init_schema(conn: &mut Connection) -> Result<()> {
        let sql = r#"
        CREATE TABLE IF NOT EXISTS documents (
            doc_id              TEXT PRIMARY KEY NOT NULL,
            path                TEXT NOT NULL UNIQUE,
            title               TEXT NOT NULL,
            frontmatter_json    TEXT,
            word_count          INTEGER NOT NULL DEFAULT 0,
            mtime               INTEGER NOT NULL,
            hash                TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sections (
            section_id          TEXT PRIMARY KEY NOT NULL,
            doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            heading             TEXT NOT NULL,
            level               INTEGER NOT NULL,
            parent_id           TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
            breadcrumbs         TEXT NOT NULL,
            line_start          INTEGER NOT NULL,
            line_end            INTEGER NOT NULL,
            content             TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS links (
            link_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            source_doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            target_note         TEXT NOT NULL,
            target_heading      TEXT,
            target_block        TEXT,
            alias               TEXT,
            link_type           TEXT NOT NULL,
            line_number         INTEGER NOT NULL,
            context_snippet     TEXT,
            source_section_id   TEXT REFERENCES sections(section_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS tags (
            tag_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            tag                 TEXT NOT NULL,
            section_id          TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
            line_number         INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id            TEXT PRIMARY KEY NOT NULL,
            doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            section_id          TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
            title               TEXT NOT NULL,
            breadcrumbs         TEXT NOT NULL,
            line_start          INTEGER NOT NULL,
            line_end            INTEGER NOT NULL,
            content             TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            breadcrumbs,
            content,
            content='chunks',
            content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, chunk_id, title, breadcrumbs, content)
            VALUES (new.rowid, new.chunk_id, new.title, new.breadcrumbs, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, title, breadcrumbs, content)
            VALUES ('delete', old.rowid, old.chunk_id, old.title, old.breadcrumbs, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, title, breadcrumbs, content)
            VALUES ('delete', old.rowid, old.chunk_id, old.title, old.breadcrumbs, old.content);
            INSERT INTO chunks_fts(rowid, chunk_id, title, breadcrumbs, content)
            VALUES (new.rowid, new.chunk_id, new.title, new.breadcrumbs, new.content);
        END;

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id            TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding           BLOB NOT NULL,
            dimensions          INTEGER NOT NULL,
            model               TEXT NOT NULL,
            created_at          INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_title          TEXT PRIMARY KEY NOT NULL,
            embedding           BLOB NOT NULL,
            dimensions          INTEGER NOT NULL,
            model               TEXT NOT NULL,
            updated_at          INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        CREATE TABLE IF NOT EXISTS sync_manifest (
            path                TEXT PRIMARY KEY NOT NULL,
            doc_id              TEXT NOT NULL,
            mtime               INTEGER NOT NULL,
            file_size           INTEGER NOT NULL,
            hash                TEXT NOT NULL,
            synced_at           INTEGER NOT NULL,
            status              TEXT NOT NULL DEFAULT 'synced'
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_mtime ON documents(mtime);

        CREATE INDEX IF NOT EXISTS idx_sections_doc_id ON sections(doc_id);
        CREATE INDEX IF NOT EXISTS idx_sections_parent_id ON sections(parent_id);

        CREATE INDEX IF NOT EXISTS idx_links_source_doc ON links(source_doc_id);
        CREATE INDEX IF NOT EXISTS idx_links_target_note ON links(target_note);
        CREATE INDEX IF NOT EXISTS idx_links_compound ON links(target_note, target_heading);
        CREATE INDEX IF NOT EXISTS idx_links_source_section ON links(source_section_id);

        CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
        CREATE INDEX IF NOT EXISTS idx_tags_doc_id ON tags(doc_id);
        CREATE INDEX IF NOT EXISTS idx_tags_compound ON tags(tag, doc_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);

        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON chunk_embeddings(model);
        CREATE INDEX IF NOT EXISTS idx_node_embeddings_model ON node_embeddings(model);
        "#;

        conn.execute_batch(sql)?;
        Ok(())
    }

    /// Atomically synchronizes a single parsed document.
    pub fn sync_document(
        &self,
        doc: &ParsedDocument,
        doc_id: &str,
        rel_path: &str,
        mtime_ms: u64,
        file_size: u64,
        embeddings: Option<&[(String, Vec<f32>, String)]>,
    ) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;

        Self::sync_document_in_tx(&tx, doc, doc_id, rel_path, mtime_ms, file_size, embeddings)?;

        tx.commit()?;
        Ok(())
    }

    /// Synchronizes document entities within an existing transaction.
    pub fn sync_document_in_tx(
        tx: &Transaction,
        doc: &ParsedDocument,
        doc_id: &str,
        rel_path: &str,
        mtime_ms: u64,
        file_size: u64,
        embeddings: Option<&[(String, Vec<f32>, String)]>,
    ) -> Result<()> {
        tx.execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;

        let frontmatter_json = serde_json::to_string(&doc.frontmatter)?;
        tx.execute(
            r#"INSERT INTO documents (doc_id, path, title, frontmatter_json, word_count, mtime, hash)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
            params![
                doc_id,
                rel_path,
                doc.title,
                frontmatter_json,
                doc.word_count as i64,
                mtime_ms as i64,
                doc.content_hash,
            ],
        )?;

        {
            let mut stmt = tx.prepare_cached(
                r#"INSERT INTO sections (section_id, doc_id, heading, level, parent_id, breadcrumbs, line_start, line_end, content)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
            )?;
            for sec in &doc.sections {
                let breadcrumbs_json = serde_json::to_string(&sec.breadcrumbs)?;
                stmt.execute(params![
                    sec.id,
                    doc_id,
                    sec.heading,
                    sec.level as i32,
                    sec.parent_id,
                    breadcrumbs_json,
                    sec.line_start as i64,
                    sec.line_end as i64,
                    sec.content,
                ])?;
            }
        }

        {
            let mut stmt = tx.prepare_cached(
                r#"INSERT INTO links (source_doc_id, target_note, target_heading, target_block, alias, link_type, line_number, context_snippet, source_section_id)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
            )?;
            for link in &doc.links {
                let link_type_str = match link.link_type {
                    LinkType::Wikilink => "wikilink",
                    LinkType::Embed => "embed",
                    LinkType::Markdown => "markdown",
                };
                stmt.execute(params![
                    doc_id,
                    link.target_note,
                    link.target_heading,
                    link.target_block,
                    link.alias,
                    link_type_str,
                    link.line_number as i64,
                    link.context_snippet,
                    link.source_section_id,
                ])?;
            }
        }

        {
            let mut stmt = tx.prepare_cached(
                r#"INSERT INTO tags (doc_id, tag, section_id, line_number)
                   VALUES (?1, ?2, ?3, ?4)"#,
            )?;
            for tag in &doc.tags {
                stmt.execute(params![
                    doc_id,
                    tag.name,
                    tag.section_id,
                    tag.line_number as i64,
                ])?;
            }
        }

        {
            let mut stmt = tx.prepare_cached(
                r#"INSERT INTO chunks (chunk_id, doc_id, section_id, title, breadcrumbs, line_start, line_end, content)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
            )?;
            for chunk in &doc.chunks {
                stmt.execute(params![
                    chunk.chunk_id,
                    doc_id,
                    chunk.section_id,
                    chunk.title,
                    chunk.breadcrumbs,
                    chunk.line_start as i64,
                    chunk.line_end as i64,
                    chunk.content,
                ])?;
            }
        }

        if let Some(embed_list) = embeddings {
            let mut stmt = tx.prepare_cached(
                r#"INSERT INTO chunk_embeddings (chunk_id, embedding, dimensions, model)
                   VALUES (?1, ?2, ?3, ?4)"#,
            )?;
            for (chunk_id, vec, model) in embed_list {
                let blob = Self::f32_slice_to_bytes(vec);
                stmt.execute(params![
                    chunk_id,
                    blob,
                    vec.len() as i64,
                    model,
                ])?;
            }
        }

        let now_sec = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs() as i64;

        tx.execute(
            r#"INSERT INTO sync_manifest (path, doc_id, mtime, file_size, hash, synced_at, status)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'synced')
               ON CONFLICT(path) DO UPDATE SET
                   doc_id = excluded.doc_id,
                   mtime = excluded.mtime,
                   file_size = excluded.file_size,
                   hash = excluded.hash,
                   synced_at = excluded.synced_at,
                   status = 'synced'"#,
            params![rel_path, doc_id, mtime_ms as i64, file_size as i64, doc.content_hash, now_sec],
        )?;

        Ok(())
    }

    /// Atomically deletes a document and all related rows.
    pub fn delete_document(&self, doc_id: &str, rel_path: &str) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;

        tx.execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;
        tx.execute("DELETE FROM sync_manifest WHERE path = ?1", params![rel_path])?;

        tx.commit()?;
        Ok(())
    }

    /// Purges all orphaned documents not present in active_paths.
    pub fn purge_orphans(&self, active_paths: &HashSet<String>) -> Result<usize> {
        let mut writer = self.writer.lock().unwrap();
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let orphans: Vec<(String, String)> = {
            let mut stmt = tx.prepare("SELECT path, doc_id FROM sync_manifest")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            let mut list = Vec::new();
            for item in rows {
                let (path, doc_id) = item?;
                if !active_paths.contains(&path) {
                    list.push((path, doc_id));
                }
            }
            list
        };

        let orphan_count = orphans.len();
        for (path, doc_id) in orphans {
            tx.execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;
            tx.execute("DELETE FROM sync_manifest WHERE path = ?1", params![path])?;
        }

        tx.commit()?;
        Ok(orphan_count)
    }

    /// Retrieve stored manifest entry for a relative path: (hash, mtime, size).
    pub fn get_manifest(&self, rel_path: &str) -> Result<Option<(String, u64, u64)>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT hash, mtime, file_size FROM sync_manifest WHERE path = ?1",
            )?;
            let mut rows = stmt.query(params![rel_path])?;
            if let Some(row) = rows.next()? {
                let hash: String = row.get(0)?;
                let mtime: i64 = row.get(1)?;
                let size: i64 = row.get(2)?;
                Ok(Some((hash, mtime as u64, size as u64)))
            } else {
                Ok(None)
            }
        })
    }

    /// Convert f32 slice to bytes.
    pub fn f32_slice_to_bytes(slice: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(slice.len() * 4);
        for &val in slice {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Convert byte slice to f32 vector.
    pub fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, StorageError> {
        if bytes.len() % 4 != 0 {
            return Err(StorageError::InvalidEmbeddingLength(bytes.len()));
        }
        let mut vec = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            vec.push(f32::from_le_bytes(arr));
        }
        Ok(vec)
    }

    /// In-memory parallel cosine similarity search over stored embeddings using Rayon.
    pub fn search_vectors(
        &self,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare("SELECT chunk_id, embedding FROM chunk_embeddings")?;
            let raw_rows = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })?;

            let mut records: Vec<(String, Vec<u8>)> = Vec::new();
            for row in raw_rows {
                records.push(row?);
            }

            let mut scored: Vec<(String, f32)> = records
                .into_par_iter()
                .filter_map(|(id, blob)| {
                    if blob.len() != query_vector.len() * 4 {
                        return None;
                    }
                    let dim = query_vector.len();
                    let mut dot = 0.0f32;
                    for i in 0..dim {
                        let off = i * 4;
                        let val = f32::from_le_bytes([
                            blob[off],
                            blob[off + 1],
                            blob[off + 2],
                            blob[off + 3],
                        ]);
                        dot += val * query_vector[i];
                    }
                    Some((id, dot))
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);
            Ok(scored)
        })
    }

    /// Saves node embeddings (e.g. Node2Vec graph embeddings) to SQLite.
    pub fn save_node_embeddings(
        &self,
        embeddings: &[(String, Vec<f32>)],
        model: &str,
    ) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let mut stmt = tx.prepare_cached(
            r#"INSERT INTO node_embeddings (node_title, embedding, dimensions, model, updated_at)
               VALUES (?1, ?2, ?3, ?4, strftime('%s', 'now'))
               ON CONFLICT(node_title) DO UPDATE SET
                   embedding = excluded.embedding,
                   dimensions = excluded.dimensions,
                   model = excluded.model,
                   updated_at = excluded.updated_at"#,
        )?;

        for (title, vec) in embeddings {
            let blob = Self::f32_slice_to_bytes(vec);
            stmt.execute(params![
                title,
                blob,
                vec.len() as i64,
                model,
            ])?;
        }

        drop(stmt);
        tx.commit()?;
        Ok(())
    }

    /// Loads all node embeddings for a given model from SQLite.
    pub fn load_node_embeddings(&self, model: &str) -> Result<Vec<(String, Vec<f32>)>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare("SELECT node_title, embedding FROM node_embeddings WHERE model = ?1")?;
            let rows = stmt.query_map(params![model], |row| {
                let title: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((title, blob))
            })?;

            let mut results = Vec::new();
            for r in rows {
                let (title, blob) = r?;
                let vec = Self::bytes_to_f32_vec(&blob)?;
                results.push((title, vec));
            }
            Ok(results)
        })
    }

    /// Fetches the embedding for a single node title.
    pub fn get_node_embedding(&self, note_title: &str, model: &str) -> Result<Option<Vec<f32>>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare("SELECT embedding FROM node_embeddings WHERE node_title = ?1 AND model = ?2")?;
            let mut rows = stmt.query(params![note_title, model])?;
            if let Some(row) = rows.next()? {
                let blob: Vec<u8> = row.get(0)?;
                let vec = Self::bytes_to_f32_vec(&blob)?;
                Ok(Some(vec))
            } else {
                Ok(None)
            }
        })
    }

    /// Full-text BM25 search over chunks with Porter stemming and diacritic removal.
    pub fn search_fts(
        &self,
        fts_query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, String, String, f64)>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT chunk_id, title, breadcrumbs, snippet(chunks_fts, 3, '<b>', '</b>', '...', 15),
                          bm25(chunks_fts, 5.0, 2.0, 1.0) as score
                   FROM chunks_fts
                   WHERE chunks_fts MATCH ?1
                   ORDER BY score ASC
                   LIMIT ?2"#,
            )?;

            let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })?;

            let mut results = Vec::new();
            for r in rows {
                results.push(r?);
            }
            Ok(results)
        })
    }

    /// Query incoming backlinks for a given note title or path.
    pub fn get_backlinks(&self, note_title: &str) -> Result<Vec<BacklinkRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT d.doc_id, d.path, d.title, l.line_number, l.link_type,
                          l.target_heading, l.target_block, l.alias, l.context_snippet
                   FROM links l
                   JOIN documents d ON l.source_doc_id = d.doc_id
                   WHERE LOWER(l.target_note) = LOWER(?1) OR l.target_note = ?1
                   ORDER BY d.title, l.line_number"#,
            )?;

            let rows = stmt.query_map(params![note_title], |row| {
                Ok(BacklinkRecord {
                    source_doc_id: row.get(0)?,
                    source_path: row.get(1)?,
                    source_title: row.get(2)?,
                    line_number: row.get::<_, i64>(3)? as usize,
                    link_type: row.get(4)?,
                    target_heading: row.get(5)?,
                    target_block: row.get(6)?,
                    alias: row.get(7)?,
                    snippet: row.get(8)?,
                })
            })?;

            let mut backlinks = Vec::new();
            for r in rows {
                backlinks.push(r?);
            }
            Ok(backlinks)
        })
    }

    /// List unresolved/dangling wikilinks sorted by reference count.
    pub fn get_unresolved_links(&self) -> Result<Vec<UnresolvedLinkRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT l.target_note, COUNT(DISTINCT l.source_doc_id) as ref_count,
                          GROUP_CONCAT(DISTINCT d.title) as sources
                   FROM links l
                   JOIN documents d ON l.source_doc_id = d.doc_id
                   WHERE l.target_note NOT IN (SELECT title FROM documents)
                     AND l.target_note NOT IN (SELECT path FROM documents)
                   GROUP BY LOWER(l.target_note)
                   ORDER BY ref_count DESC"#,
            )?;

            let rows = stmt.query_map([], |row| {
                let target: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                let sources_str: String = row.get(2)?;
                let sources = sources_str.split(',').map(|s| s.trim().to_string()).collect();
                Ok(UnresolvedLinkRecord {
                    target_note: target,
                    reference_count: count as usize,
                    source_notes: sources,
                })
            })?;

            let mut list = Vec::new();
            for r in rows {
                list.push(r?);
            }
            Ok(list)
        })
    }

    /// Get orphan notes (no incoming links and no outgoing links).
    pub fn get_orphan_notes(&self) -> Result<Vec<(String, String)>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT d.title, d.path
                   FROM documents d
                   WHERE d.doc_id NOT IN (SELECT source_doc_id FROM links)
                     AND d.title NOT IN (SELECT target_note FROM links)
                     AND d.path NOT IN (SELECT target_note FROM links)
                   ORDER BY d.title"#,
            )?;

            let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            let mut list = Vec::new();
            for r in rows {
                list.push(r?);
            }
            Ok(list)
        })
    }

    /// Returns general metrics about the indexed vault.
    pub fn get_stats(&self) -> Result<VaultStats> {
        self.with_read_conn(|conn| {
            let doc_count: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
            let sec_count: i64 = conn.query_row("SELECT COUNT(*) FROM sections", [], |r| r.get(0))?;
            let link_count: i64 = conn.query_row("SELECT COUNT(*) FROM links", [], |r| r.get(0))?;
            let tag_count: i64 = conn.query_row("SELECT COUNT(*) FROM tags", [], |r| r.get(0))?;
            let chunk_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
            let embed_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |r| r.get(0))?;
            let word_sum: i64 = conn.query_row("SELECT COALESCE(SUM(word_count), 0) FROM documents", [], |r| r.get(0))?;

            Ok(VaultStats {
                total_documents: doc_count as usize,
                total_sections: sec_count as usize,
                total_links: link_count as usize,
                total_tags: tag_count as usize,
                total_chunks: chunk_count as usize,
                total_embeddings: embed_count as usize,
                total_words: word_sum as usize,
            })
        })
    }

    /// Retrieve a single document by doc_id, path, or title.
    pub fn get_document(&self, query: &str) -> Result<Option<DocumentRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT doc_id, path, title, frontmatter_json, word_count, mtime, hash
                   FROM documents
                   WHERE doc_id = ?1 OR path = ?1 OR LOWER(title) = LOWER(?1)
                   LIMIT 1"#,
            )?;

            let mut rows = stmt.query(params![query])?;
            if let Some(row) = rows.next()? {
                Ok(Some(DocumentRecord {
                    doc_id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    frontmatter_json: row.get(3)?,
                    word_count: row.get::<_, i64>(4)? as usize,
                    mtime: row.get::<_, i64>(5)? as u64,
                    hash: row.get(6)?,
                }))
            } else {
                Ok(None)
            }
        })
    }

    /// Retrieve all chunks belonging to a document.
    pub fn get_chunks_for_doc(&self, doc_id: &str) -> Result<Vec<ChunkRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT chunk_id, doc_id, section_id, title, breadcrumbs, line_start, line_end, content
                   FROM chunks
                   WHERE doc_id = ?1
                   ORDER BY line_start ASC"#,
            )?;

            let rows = stmt.query_map(params![doc_id], |row| {
                let breadcrumbs_raw: String = row.get(4)?;
                let breadcrumbs: Vec<String> = serde_json::from_str(&breadcrumbs_raw).unwrap_or_default();
                Ok(ChunkRecord {
                    chunk_id: row.get(0)?,
                    doc_id: row.get(1)?,
                    section_id: row.get(2)?,
                    title: row.get(3)?,
                    breadcrumbs,
                    line_start: row.get::<_, i64>(5)? as usize,
                    line_end: row.get::<_, i64>(6)? as usize,
                    content: row.get(7)?,
                })
            })?;

            let mut chunks = Vec::new();
            for r in rows {
                chunks.push(r?);
            }
            Ok(chunks)
        })
    }

    /// Retrieve all sections belonging to a document.
    pub fn get_sections_for_doc(&self, doc_id: &str) -> Result<Vec<SectionRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT section_id, doc_id, heading, level, parent_id, breadcrumbs, line_start, line_end, content
                   FROM sections
                   WHERE doc_id = ?1
                   ORDER BY line_start ASC"#,
            )?;

            let rows = stmt.query_map(params![doc_id], |row| {
                let breadcrumbs_raw: String = row.get(5)?;
                let breadcrumbs: Vec<String> = serde_json::from_str(&breadcrumbs_raw).unwrap_or_default();
                Ok(SectionRecord {
                    section_id: row.get(0)?,
                    doc_id: row.get(1)?,
                    heading: row.get(2)?,
                    level: row.get::<_, i32>(3)? as usize,
                    parent_id: row.get(4)?,
                    breadcrumbs,
                    line_start: row.get::<_, i64>(6)? as usize,
                    line_end: row.get::<_, i64>(7)? as usize,
                    content: row.get(8)?,
                })
            })?;

            let mut sections = Vec::new();
            for r in rows {
                sections.push(r?);
            }
            Ok(sections)
        })
    }

    /// Retrieve all outgoing links originating from a document.
    pub fn get_outlinks_for_doc(&self, doc_id: &str) -> Result<Vec<LinkRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT source_doc_id, target_note, target_heading, target_block, alias, link_type, line_number, context_snippet
                   FROM links
                   WHERE source_doc_id = ?1
                   ORDER BY line_number ASC"#,
            )?;

            let rows = stmt.query_map(params![doc_id], |row| {
                Ok(LinkRecord {
                    source_doc_id: row.get(0)?,
                    target_note: row.get(1)?,
                    target_heading: row.get(2)?,
                    target_block: row.get(3)?,
                    alias: row.get(4)?,
                    link_type: row.get(5)?,
                    line_number: row.get::<_, i64>(6)? as usize,
                    context_snippet: row.get(7)?,
                })
            })?;

            let mut links = Vec::new();
            for r in rows {
                links.push(r?);
            }
            Ok(links)
        })
    }

    /// Retrieve all indexed document paths.
    pub fn get_all_document_paths(&self) -> Result<Vec<String>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached("SELECT path FROM documents ORDER BY path ASC")?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            let mut paths = Vec::new();
            for r in rows {
                paths.push(r?);
            }
            Ok(paths)
        })
    }

    /// Retrieve all indexed documents.
    pub fn get_all_documents(&self) -> Result<Vec<DocumentRecord>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                r#"SELECT doc_id, path, title, frontmatter_json, word_count, mtime, hash
                   FROM documents
                   ORDER BY title ASC"#,
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(DocumentRecord {
                    doc_id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    frontmatter_json: row.get(3)?,
                    word_count: row.get::<_, i64>(4)? as usize,
                    mtime: row.get::<_, i64>(5)? as u64,
                    hash: row.get(6)?,
                })
            })?;

            let mut docs = Vec::new();
            for r in rows {
                docs.push(r?);
            }
            Ok(docs)
        })
    }

    /// Retrieve all distinct tags indexed across the vault.
    pub fn get_all_tags(&self) -> Result<Vec<String>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached("SELECT DISTINCT tag FROM tags ORDER BY tag ASC")?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            let mut tags = Vec::new();
            for r in rows {
                tags.push(r?);
            }
            Ok(tags)
        })
    }

    /// Get count of distinct documents containing a given tag.
    pub fn get_tag_count(&self, tag: &str) -> Result<usize> {
        self.with_read_conn(|conn| {
            let clean_tag = tag.trim_start_matches('#');
            let mut stmt = conn.prepare_cached(
                "SELECT COUNT(DISTINCT doc_id) FROM tags WHERE tag = ?1 OR tag = ?2",
            )?;
            let with_hash = format!("#{clean_tag}");
            let count: i64 = stmt.query_row(params![clean_tag, with_hash], |r| r.get(0))?;
            Ok(count as usize)
        })
    }
    /// Retrieve all node embeddings stored in the database.
    pub fn get_all_node_embeddings(&self) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        self.with_read_conn(|conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT node_title, embedding FROM node_embeddings ORDER BY node_title ASC",
            )?;
            let rows = stmt.query_map([], |row| {
                let title: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Ok((title, floats))
            })?;

            let mut map = std::collections::HashMap::new();
            for r in rows {
                let (title, floats) = r?;
                map.insert(title, floats);
            }
            Ok(map)
        })
    }
}
