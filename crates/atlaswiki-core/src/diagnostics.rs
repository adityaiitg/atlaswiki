//! Link Diagnostics & Typo Correction Engine for AtlasWiki.
//!
//! Validates wikilinks, section anchors (#heading), and block references (^block-id).
//! Employs Damerau-Levenshtein distance to suggest corrections for broken links.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use atlaswiki_parser::ast::{LinkType, ParsedDocument};
use atlaswiki_parser::slugify;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Warning,
    Error,
    Info,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticCode {
    UnresolvedWikilink,
    BrokenHeadingAnchor,
    BrokenBlockReference,
    UnresolvedEmbed,
    DeadMarkdownLink,
}

impl DiagnosticCode {
    pub fn code_str(&self) -> &'static str {
        match self {
            DiagnosticCode::UnresolvedWikilink => "W001",
            DiagnosticCode::BrokenHeadingAnchor => "W002",
            DiagnosticCode::BrokenBlockReference => "W003",
            DiagnosticCode::UnresolvedEmbed => "W004",
            DiagnosticCode::DeadMarkdownLink => "W005",
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            DiagnosticCode::UnresolvedWikilink => "unresolved wikilink",
            DiagnosticCode::BrokenHeadingAnchor => "broken heading anchor",
            DiagnosticCode::BrokenBlockReference => "broken block reference",
            DiagnosticCode::UnresolvedEmbed => "unresolved embed target",
            DiagnosticCode::DeadMarkdownLink => "dead markdown link",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLocation {
    pub file_path: PathBuf,
    pub line_number: usize,
    pub col_start: usize,
    pub col_end: usize,
    pub source_line: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub location: DiagnosticLocation,
    pub suggestion: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WantedPage {
    pub target: String,
    pub reference_count: usize,
    pub referencing_files: Vec<PathBuf>,
    pub occurrences: Vec<DiagnosticLocation>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsReport {
    pub total_files_scanned: usize,
    pub total_links_checked: usize,
    pub diagnostics: Vec<Diagnostic>,
    pub wanted_pages: Vec<WantedPage>,
}

impl DiagnosticsReport {
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Warning)
            .count()
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .count()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Damerau-Levenshtein typo correction engine with transposition support and fast pruning.
#[derive(Debug, Clone)]
pub struct TypoCorrectionEngine {
    pub max_edit_distance: usize,
    pub min_similarity_ratio: f64,
}

impl Default for TypoCorrectionEngine {
    fn default() -> Self {
        Self {
            max_edit_distance: 3,
            min_similarity_ratio: 0.60,
        }
    }
}

impl TypoCorrectionEngine {
    pub fn new(max_edit_distance: usize, min_similarity_ratio: f64) -> Self {
        Self {
            max_edit_distance,
            min_similarity_ratio,
        }
    }

    pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut d = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m {
            d[i][0] = i;
        }
        for j in 0..=n {
            d[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i - 1].to_ascii_lowercase()
                    == b_chars[j - 1].to_ascii_lowercase()
                {
                    0
                } else {
                    1
                };

                d[i][j] = (d[i - 1][j] + 1)
                    .min(d[i][j - 1] + 1)
                    .min(d[i - 1][j - 1] + cost);

                if i > 1
                    && j > 1
                    && a_chars[i - 1].to_ascii_lowercase()
                        == b_chars[j - 2].to_ascii_lowercase()
                    && a_chars[i - 2].to_ascii_lowercase()
                        == b_chars[j - 1].to_ascii_lowercase()
                {
                    d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
                }
            }
        }

        d[m][n]
    }

    pub fn suggest<'a>(
        &self,
        query: &str,
        candidates: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let query_len = query.chars().count();
        if query_len == 0 {
            return Vec::new();
        }

        let max_dist = if query_len <= 3 {
            1
        } else if query_len <= 7 {
            2.min(self.max_edit_distance)
        } else {
            self.max_edit_distance
        };

        let mut matches = Vec::new();

        for candidate in candidates {
            if candidate.is_empty() || candidate.eq_ignore_ascii_case(query) {
                continue;
            }

            let cand_len = candidate.chars().count();
            if cand_len.abs_diff(query_len) > max_dist {
                continue;
            }

            let dist = Self::damerau_levenshtein(query, candidate);
            let max_l = query_len.max(cand_len);
            let sim = 1.0 - (dist as f64 / max_l as f64);

            if dist <= max_dist && sim >= self.min_similarity_ratio {
                matches.push((dist, sim, candidate));
            }
        }

        matches.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.2.cmp(b.2))
        });

        let mut seen = HashSet::new();
        let mut results = Vec::new();

        for (_, _, cand) in matches {
            if seen.insert(cand.to_lowercase()) {
                results.push(cand.to_string());
                if results.len() >= 3 {
                    break;
                }
            }
        }

        results
    }
}

/// Metadata stored per note for fast diagnostic lookups.
#[derive(Debug, Clone)]
struct NoteDiagnosticsMeta {
    #[allow(dead_code)]
    file_path: PathBuf,
    title: String,
    headings: Vec<String>,
    block_ids: HashSet<String>,
}

/// Diagnostics Engine.
pub struct DiagnosticsEngine {
    typo_engine: TypoCorrectionEngine,
    notes: HashMap<PathBuf, NoteDiagnosticsMeta>,
    title_to_path: HashMap<String, PathBuf>,
    alias_to_path: HashMap<String, PathBuf>,
    stem_to_path: HashMap<String, PathBuf>,
}

impl Default for DiagnosticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticsEngine {
    pub fn new() -> Self {
        Self {
            typo_engine: TypoCorrectionEngine::default(),
            notes: HashMap::new(),
            title_to_path: HashMap::new(),
            alias_to_path: HashMap::new(),
            stem_to_path: HashMap::new(),
        }
    }

    pub fn index_document(&mut self, doc: &ParsedDocument) {
        let file_stem = doc
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut block_ids: HashSet<String> = HashSet::new();
        for sec in &doc.sections {
            for line in sec.content.lines() {
                if let Some(idx) = line.rfind('^') {
                    let pid = line[idx + 1..].trim();
                    if !pid.is_empty()
                        && pid
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                    {
                        block_ids.insert(pid.to_string());
                    }
                }
            }
        }

        let headings: Vec<String> = doc.sections.iter().map(|s| s.heading.clone()).collect();
        let meta = NoteDiagnosticsMeta {
            file_path: doc.path.clone(),
            title: doc.title.clone(),
            headings,
            block_ids,
        };

        self.stem_to_path.insert(file_stem.to_lowercase(), doc.path.clone());
        self.title_to_path.insert(doc.title.to_lowercase(), doc.path.clone());

        for alias in &doc.frontmatter.aliases {
            self.alias_to_path.insert(alias.to_lowercase(), doc.path.clone());
        }

        self.notes.insert(doc.path.clone(), meta);
    }

    pub fn resolve_target(&self, from_doc_path: Option<&std::path::Path>, target: &str) -> Option<&PathBuf> {
        let clean = target.trim_end_matches(".md").trim();
        if clean.is_empty() {
            return None;
        }

        let p = PathBuf::from(clean);
        let normalized = if let Ok(stripped) = p.strip_prefix("./") {
            stripped.to_path_buf()
        } else {
            p.clone()
        };

        if let Some((path, _)) = self.notes.get_key_value(&normalized) {
            return Some(path);
        }

        let with_md = PathBuf::from(format!("{}.md", normalized.to_string_lossy()));
        if let Some((path, _)) = self.notes.get_key_value(&with_md) {
            return Some(path);
        }

        // Relative to source file directory
        if let Some(source_path) = from_doc_path {
            if let Some(parent) = source_path.parent() {
                let rel = parent.join(&normalized);
                if let Some((path, _)) = self.notes.get_key_value(&rel) {
                    return Some(path);
                }
                let rel_md = parent.join(&with_md);
                if let Some((path, _)) = self.notes.get_key_value(&rel_md) {
                    return Some(path);
                }
            }
        }

        let lower = clean.to_lowercase();
        if let Some(path) = self.title_to_path.get(&lower) {
            return Some(path);
        }
        if let Some(path) = self.alias_to_path.get(&lower) {
            return Some(path);
        }
        if let Some(path) = self.stem_to_path.get(&lower) {
            return Some(path);
        }

        None
    }

    pub fn all_known_targets(&self) -> Vec<String> {
        let mut targets = HashSet::new();
        for meta in self.notes.values() {
            targets.insert(meta.title.clone());
        }
        for alias in self.alias_to_path.keys() {
            targets.insert(alias.clone());
        }
        targets.into_iter().collect()
    }

    pub fn run(&self, docs: &[ParsedDocument], strict_mode: bool) -> DiagnosticsReport {
        let mut diagnostics = Vec::new();
        let mut wanted_map: HashMap<String, (usize, HashSet<PathBuf>, Vec<DiagnosticLocation>, Vec<String>)> = HashMap::new();
        let mut total_links = 0;
        let all_targets = self.all_known_targets();

        for doc in docs {
            for link in &doc.links {
                total_links += 1;
                let is_self_link = link.target_note.trim().is_empty();
                let resolved_path = if is_self_link {
                    Some(&doc.path)
                } else {
                    self.resolve_target(Some(&doc.path), &link.target_note)
                };

                let location = DiagnosticLocation {
                    file_path: doc.path.clone(),
                    line_number: link.line_number,
                    col_start: 1,
                    col_end: link.target_note.len() + 4,
                    source_line: link.context_snippet.clone().unwrap_or_default(),
                };

                if resolved_path.is_none() {
                    let suggestions = self.typo_engine.suggest(
                        &link.target_note,
                        all_targets.iter().map(|s| s.as_str()),
                    );
                    let primary_suggestion = suggestions.first().map(|s| format!("[[{s}]]"));

                    let diag_code = match link.link_type {
                        LinkType::Wikilink => DiagnosticCode::UnresolvedWikilink,
                        LinkType::Embed => DiagnosticCode::UnresolvedEmbed,
                        LinkType::Markdown => DiagnosticCode::DeadMarkdownLink,
                    };

                    let severity = if strict_mode {
                        DiagnosticSeverity::Error
                    } else {
                        DiagnosticSeverity::Warning
                    };

                    diagnostics.push(Diagnostic {
                        code: diag_code,
                        severity,
                        message: format!("target note '{}' does not exist in vault", link.target_note),
                        location: location.clone(),
                        suggestion: primary_suggestion,
                        notes: Vec::new(),
                    });

                    let entry = wanted_map
                        .entry(link.target_note.clone())
                        .or_insert_with(|| (0, HashSet::new(), Vec::new(), suggestions.clone()));
                    entry.0 += 1;
                    entry.1.insert(doc.path.clone());
                    entry.2.push(location.clone());
                    continue;
                }

                let target_path = resolved_path.unwrap();
                let target_meta = self.notes.get(target_path).unwrap();

                // Broken heading check
                if let Some(ref heading_target) = link.target_heading {
                    let clean_h = heading_target.trim().trim_start_matches('#');
                    let clean_slug = slugify(clean_h);
                    let exists = target_meta
                        .headings
                        .iter()
                        .any(|h| h.eq_ignore_ascii_case(clean_h) || slugify(h) == clean_slug);

                    if !exists {
                        let suggestions = self.typo_engine.suggest(
                            clean_h,
                            target_meta.headings.iter().map(|s| s.as_str()),
                        );
                        let sug = suggestions.first().map(|s| {
                            if is_self_link {
                                format!("[[#{s}]]")
                            } else {
                                format!("[[{}#{s}]]", link.target_note)
                            }
                        });

                        diagnostics.push(Diagnostic {
                            code: DiagnosticCode::BrokenHeadingAnchor,
                            severity: DiagnosticSeverity::Warning,
                            message: format!("heading anchor '#{clean_h}' not found in note '{}'", target_meta.title),
                            location: location.clone(),
                            suggestion: sug,
                            notes: vec![format!("available headings: {}", target_meta.headings.join(", "))],
                        });
                    }
                }

                // Broken block check
                if let Some(ref block_target) = link.target_block {
                    let clean_b = block_target.trim().trim_start_matches('^');
                    let exists = target_meta.block_ids.contains(clean_b);

                    if !exists {
                        let suggestions = self.typo_engine.suggest(
                            clean_b,
                            target_meta.block_ids.iter().map(|s| s.as_str()),
                        );
                        let sug = suggestions.first().map(|s| {
                            if is_self_link {
                                format!("[[#^{s}]]")
                            } else {
                                format!("[[{}#^{s}]]", link.target_note)
                            }
                        });

                        diagnostics.push(Diagnostic {
                            code: DiagnosticCode::BrokenBlockReference,
                            severity: DiagnosticSeverity::Warning,
                            message: format!("block reference '^{clean_b}' not found in note '{}'", target_meta.title),
                            location: location.clone(),
                            suggestion: sug,
                            notes: vec![format!("existing block ids: {}", target_meta.block_ids.iter().cloned().collect::<Vec<_>>().join(", "))],
                        });
                    }
                }
            }
        }

        let mut wanted_pages: Vec<WantedPage> = wanted_map
            .into_iter()
            .map(|(target, (count, files, occs, sugs))| WantedPage {
                target,
                reference_count: count,
                referencing_files: files.into_iter().collect(),
                occurrences: occs,
                suggestions: sugs,
            })
            .collect();

        wanted_pages.sort_by(|a, b| b.reference_count.cmp(&a.reference_count));

        DiagnosticsReport {
            total_files_scanned: docs.len(),
            total_links_checked: total_links,
            diagnostics,
            wanted_pages,
        }
    }
}
