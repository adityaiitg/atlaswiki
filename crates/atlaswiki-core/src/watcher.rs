//! Real-time filesystem watcher & incremental sync engine for AtlasWiki.
//!
//! Provides sliding-window debouncing, atomic editor save detection,
//! three-tier change verification, SQLite synchronization, and rename tracking.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::Result;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use sha2::{Digest, Sha256};

use atlaswiki_parser::MarkdownParser;
use crate::storage::StorageEngine;

/// Configuration parameters for the VaultWatcher.
#[derive(Debug, Clone)]
pub struct VaultWatcherConfig {
    pub vault_root: PathBuf,
    pub debounce_duration: Duration,
    pub tick_interval: Duration,
    pub ignored_directories: Vec<String>,
}

impl Default for VaultWatcherConfig {
    fn default() -> Self {
        Self {
            vault_root: PathBuf::from("."),
            debounce_duration: Duration::from_millis(600),
            tick_interval: Duration::from_millis(50),
            ignored_directories: vec![
                ".git".to_string(),
                ".obsidian".to_string(),
                ".trash".to_string(),
                ".atlaswiki".to_string(),
                ".stversions".to_string(),
            ],
        }
    }
}

/// A reference to a wikilink broken due to a note rename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokenLinkRef {
    pub source_note: String,
    pub line_number: usize,
    pub old_target: String,
    pub new_target: String,
    pub heading: Option<String>,
    pub alias: Option<String>,
    pub context_snippet: Option<String>,
}

/// Report summarizing a rename operation and affected links.
#[derive(Debug, Clone)]
pub struct RenameReport {
    pub old_path: PathBuf,
    pub new_path: PathBuf,
    pub old_stem: String,
    pub new_stem: String,
    pub broken_links: Vec<BrokenLinkRef>,
}

/// Events broadcast to external listeners (CLI, UI, MCP).
#[derive(Debug, Clone)]
pub enum VaultSyncEvent {
    NoteIndexed {
        path: PathBuf,
        word_count: usize,
        chunks_count: usize,
    },
    NoteDeleted {
        path: PathBuf,
    },
    NoteRenamed(RenameReport),
    SyncError {
        path: PathBuf,
        error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingActionKind {
    Created,
    Modified,
    Deleted,
}

#[derive(Debug, Clone)]
struct PendingEntry {
    kind: PendingActionKind,
    last_event_at: Instant,
}

struct DebounceState {
    pending: HashMap<PathBuf, PendingEntry>,
    duration: Duration,
}

impl DebounceState {
    fn new(duration: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            duration,
        }
    }

    fn record_event(&mut self, path: PathBuf, kind: PendingActionKind) {
        let now = Instant::now();
        if let Some(existing) = self.pending.get_mut(&path) {
            existing.last_event_at = now;
            match (existing.kind, kind) {
                // Delete followed by create within debounce window = atomic editor replacement
                (PendingActionKind::Deleted, PendingActionKind::Created) => {
                    existing.kind = PendingActionKind::Modified;
                }
                // Created followed by modify = still created
                (PendingActionKind::Created, PendingActionKind::Modified) => {
                    existing.kind = PendingActionKind::Created;
                }
                // Created then deleted = cancel both
                (PendingActionKind::Created, PendingActionKind::Deleted) => {
                    self.pending.remove(&path);
                }
                // Modified then deleted = deleted
                (PendingActionKind::Modified, PendingActionKind::Deleted) => {
                    existing.kind = PendingActionKind::Deleted;
                }
                _ => {
                    existing.kind = kind;
                }
            }
        } else {
            self.pending.insert(
                path,
                PendingEntry {
                    kind,
                    last_event_at: now,
                },
            );
        }
    }

    fn drain_settled(&mut self) -> Vec<(PathBuf, PendingActionKind)> {
        let now = Instant::now();
        let mut settled = Vec::new();

        self.pending.retain(|path, entry| {
            if now.duration_since(entry.last_event_at) >= self.duration {
                settled.push((path.clone(), entry.kind));
                false
            } else {
                true
            }
        });

        settled
    }
}

/// Filesystem Watcher Engine.
pub struct VaultWatcher {
    is_running: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
}

impl VaultWatcher {
    pub fn start(
        config: VaultWatcherConfig,
        storage: StorageEngine,
    ) -> Result<(Self, Receiver<VaultSyncEvent>)> {
        let is_running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&is_running);

        let (sync_tx, sync_rx) = channel();
        let event_tx_clone = sync_tx.clone();

        let worker_handle = thread::Builder::new()
            .name("atlaswiki-vault-watcher".to_string())
            .spawn(move || {
                if let Err(e) = Self::run_loop(config, storage, running_clone, event_tx_clone) {
                    eprintln!("[VaultWatcher Error] Worker thread exited: {e:#}");
                }
            })?;

        Ok((
            Self {
                is_running,
                worker_handle: Some(worker_handle),
            },
            sync_rx,
        ))
    }

    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }

    fn run_loop(
        config: VaultWatcherConfig,
        storage: StorageEngine,
        is_running: Arc<AtomicBool>,
        event_tx: Sender<VaultSyncEvent>,
    ) -> Result<()> {
        let parser = MarkdownParser::new();
        let (fs_tx, fs_rx) = channel();

        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                if let Ok(event) = res {
                    let _ = fs_tx.send(event);
                }
            },
            Config::default(),
        )?;

        watcher.watch(&config.vault_root, RecursiveMode::Recursive)?;

        let mut debouncer = DebounceState::new(config.debounce_duration);
        let canonical_root = config
            .vault_root
            .canonicalize()
            .unwrap_or_else(|_| config.vault_root.clone());

        while is_running.load(Ordering::SeqCst) {
            while let Ok(event) = fs_rx.try_recv() {
                for path in event.paths {
                    if !Self::is_watchable_file(&path, &canonical_root, &config.ignored_directories) {
                        continue;
                    }

                    match event.kind {
                        EventKind::Create(_) => {
                            debouncer.record_event(path, PendingActionKind::Created);
                        }
                        EventKind::Modify(_) => {
                            debouncer.record_event(path, PendingActionKind::Modified);
                        }
                        EventKind::Remove(_) => {
                            debouncer.record_event(path, PendingActionKind::Deleted);
                        }
                        _ => {}
                    }
                }
            }

            let settled = debouncer.drain_settled();

            for (abs_path, action) in settled {
                let rel_path = match abs_path.strip_prefix(&canonical_root) {
                    Ok(p) => p.to_string_lossy().to_string(),
                    Err(_) => continue,
                };

                match action {
                    PendingActionKind::Deleted => {
                        let doc_id = rel_path.clone();
                        if let Err(e) = storage.delete_document(&doc_id, &rel_path) {
                            let _ = event_tx.send(VaultSyncEvent::SyncError {
                                path: abs_path,
                                error: e.to_string(),
                            });
                        } else {
                            let _ = event_tx.send(VaultSyncEvent::NoteDeleted {
                                path: PathBuf::from(&rel_path),
                            });
                        }
                    }
                    PendingActionKind::Created | PendingActionKind::Modified => {
                        Self::process_file_upsert(
                            &abs_path,
                            &rel_path,
                            &storage,
                            &parser,
                            &event_tx,
                        );
                    }
                }
            }

            thread::sleep(config.tick_interval);
        }

        Ok(())
    }

    fn process_file_upsert(
        abs_path: &Path,
        rel_path: &str,
        storage: &StorageEngine,
        parser: &MarkdownParser,
        event_tx: &Sender<VaultSyncEvent>,
    ) {
        let meta = match fs::metadata(abs_path) {
            Ok(m) => m,
            Err(_) => {
                let _ = storage.delete_document(rel_path, rel_path);
                return;
            }
        };

        let size_bytes = meta.len();
        let mtime_ms = meta
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Ok(Some((stored_hash, stored_mtime, stored_size))) = storage.get_manifest(rel_path) {
            if stored_mtime == mtime_ms && stored_size == size_bytes {
                return;
            }

            let content = match fs::read_to_string(abs_path) {
                Ok(c) => c,
                Err(e) => {
                    let _ = event_tx.send(VaultSyncEvent::SyncError {
                        path: abs_path.to_path_buf(),
                        error: format!("Failed to read file: {e}"),
                    });
                    return;
                }
            };

            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            let current_hash = format!("{:x}", hasher.finalize());

            if current_hash == stored_hash {
                return;
            }

            match parser.parse_file(Path::new(rel_path), &content) {
                Ok(doc) => {
                    let word_count = doc.word_count;
                    let chunks_count = doc.chunks.len();
                    if let Err(e) = storage.sync_document(&doc, rel_path, rel_path, mtime_ms, size_bytes, None) {
                        let _ = event_tx.send(VaultSyncEvent::SyncError {
                            path: abs_path.to_path_buf(),
                            error: format!("Database transaction failed: {e}"),
                        });
                    } else {
                        let _ = event_tx.send(VaultSyncEvent::NoteIndexed {
                            path: PathBuf::from(rel_path),
                            word_count,
                            chunks_count,
                        });
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(VaultSyncEvent::SyncError {
                        path: abs_path.to_path_buf(),
                        error: format!("Parser failed: {e}"),
                    });
                }
            }
        } else {
            let content = match fs::read_to_string(abs_path) {
                Ok(c) => c,
                Err(e) => {
                    let _ = event_tx.send(VaultSyncEvent::SyncError {
                        path: abs_path.to_path_buf(),
                        error: format!("Failed to read file: {e}"),
                    });
                    return;
                }
            };

            match parser.parse_file(Path::new(rel_path), &content) {
                Ok(doc) => {
                    let word_count = doc.word_count;
                    let chunks_count = doc.chunks.len();
                    if let Err(e) = storage.sync_document(&doc, rel_path, rel_path, mtime_ms, size_bytes, None) {
                        let _ = event_tx.send(VaultSyncEvent::SyncError {
                            path: abs_path.to_path_buf(),
                            error: format!("Database transaction failed: {e}"),
                        });
                    } else {
                        let _ = event_tx.send(VaultSyncEvent::NoteIndexed {
                            path: PathBuf::from(rel_path),
                            word_count,
                            chunks_count,
                        });
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(VaultSyncEvent::SyncError {
                        path: abs_path.to_path_buf(),
                        error: format!("Parser failed: {e}"),
                    });
                }
            }
        }
    }

    fn is_watchable_file(path: &Path, root: &Path, ignored: &[String]) -> bool {
        let is_md = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown"))
            .unwrap_or(false);

        if !is_md {
            return false;
        }

        if let Ok(rel) = path.strip_prefix(root) {
            for comp in rel.components() {
                if let std::path::Component::Normal(c) = comp {
                    let name = c.to_string_lossy();
                    if ignored.iter().any(|ig| ig == &name) {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }
}
