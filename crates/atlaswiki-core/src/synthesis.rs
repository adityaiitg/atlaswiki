//! Automated Map of Content (MOC) and Living Wiki Index Synthesis Engine.
//!
//! Provides automated Map of Content generation, tag taxonomy tree hierarchy grouping,
//! PageRank hub centrality ordering, orphan cataloging, and idempotent non-destructive
//! markdown block replacement.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};

use atlaswiki_parser::{MarkdownParser, ParsedDocument};

use crate::graph::KnowledgeGraph;

/// Marker designating the start of AtlasWiki-generated synthesized content.
pub const SYNTHESIS_BEGIN: &str = "<!-- ATLASWIKI:BEGIN_SYNTHESIS -->";

/// Marker designating the end of AtlasWiki-generated synthesized content.
pub const SYNTHESIS_END: &str = "<!-- ATLASWIKI:END_SYNTHESIS -->";

/// A summary of an indexed note for MOC synthesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoteSummary {
    pub title: String,
    pub path: String,
    pub tags: Vec<String>,
    pub word_count: usize,
    pub pagerank: f32,
    pub in_degree: usize,
    pub out_degree: usize,
}

/// A node in the hierarchical tag taxonomy tree.
#[derive(Debug, Clone, Default)]
pub struct TagTaxonomy {
    pub segment: String,
    pub full_path: String,
    pub notes: Vec<NoteSummary>,
    pub children: BTreeMap<String, TagTaxonomy>,
}

impl TagTaxonomy {
    /// Inserts a note into the taxonomy tree based on its hierarchical tag path (e.g. `ai/ml/nlp`).
    pub fn insert(&mut self, tag_path: &str, note: &NoteSummary) {
        let clean = tag_path.trim().trim_start_matches('#');
        if clean.is_empty() {
            return;
        }
        let segments: Vec<&str> = clean.split('/').filter(|s| !s.is_empty()).collect();
        self.insert_recursive(&segments, clean, note, 0);
    }

    fn insert_recursive(
        &mut self,
        segments: &[&str],
        full_path: &str,
        note: &NoteSummary,
        depth: usize,
    ) {
        if depth >= segments.len() {
            if !self.notes.iter().any(|n| n.title == note.title) {
                self.notes.push(note.clone());
            }
            return;
        }

        let segment = segments[depth].to_string();
        let sub_full_path = segments[0..=depth].join("/");
        let child = self
            .children
            .entry(segment.clone())
            .or_insert_with(|| TagTaxonomy {
                segment,
                full_path: sub_full_path,
                notes: Vec::new(),
                children: BTreeMap::new(),
            });

        child.insert_recursive(segments, full_path, note, depth + 1);
    }

    /// Recursively sorts all notes in the taxonomy by PageRank descending, then title ascending.
    pub fn sort_by_pagerank(&mut self) {
        self.notes.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.title.cmp(&b.title))
        });
        for child in self.children.values_mut() {
            child.sort_by_pagerank();
        }
    }

    /// Renders the tag taxonomy tree to Markdown.
    pub fn render_markdown(&self, depth: usize, out: &mut String) {
        for child in self.children.values() {
            let heading_level = "#".repeat((depth + 2).clamp(2, 6));
            out.push_str(&format!("{} #{}\n\n", heading_level, child.full_path));

            if !child.notes.is_empty() {
                for note in &child.notes {
                    out.push_str(&format!(
                        "- [[{}]] · Hub: {:.2} (In: {}, Out: {})\n",
                        note.title, note.pagerank, note.in_degree, note.out_degree
                    ));
                }
                out.push('\n');
            }

            child.render_markdown(depth + 1, out);
        }
    }
}

/// Metadata describing a synthesized Topic Map of Content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMocInfo {
    pub topic: String,
    pub file_name: String,
    pub title: String,
    pub note_count: usize,
    pub hub_notes: Vec<String>,
}

/// Report returned by living wiki index generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivingWikiReport {
    pub index_path: String,
    pub orphans_path: String,
    pub topic_mocs: Vec<TopicMocInfo>,
    pub total_notes: usize,
    pub total_orphans: usize,
    pub dry_run: bool,
}

/// Report returned by single MOC generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MocReport {
    pub topic: String,
    pub file_path: String,
    pub note_count: usize,
    pub hub_notes: Vec<String>,
    pub dry_run: bool,
    pub content: Option<String>,
}

/// Idempotently replaces or inserts the synthesized block into existing note content.
/// Preserves any user edits outside `<!-- ATLASWIKI:BEGIN_SYNTHESIS -->` and `<!-- ATLASWIKI:END_SYNTHESIS -->`.
pub fn update_synthesis_content(
    existing: &str,
    synthesized_body: &str,
    default_header: Option<&str>,
) -> String {
    let body = synthesized_body.trim();
    let block = format!("{}\n{}\n{}", SYNTHESIS_BEGIN, body, SYNTHESIS_END);

    if let Some(begin_idx) = existing.find(SYNTHESIS_BEGIN) {
        if let Some(end_rel) = existing[begin_idx..].find(SYNTHESIS_END) {
            let end_idx = begin_idx + end_rel + SYNTHESIS_END.len();
            let before = &existing[..begin_idx];
            let after = &existing[end_idx..];
            return format!("{}{}{}", before, block, after);
        }
    }

    if existing.trim().is_empty() {
        if let Some(header) = default_header {
            format!("{}\n\n{}\n", header.trim(), block)
        } else {
            format!("{}\n", block)
        }
    } else {
        format!("{}\n\n{}\n", existing.trim_end(), block)
    }
}

/// Sanitizes a topic string into a clean note title and file name.
pub fn sanitize_topic_name(topic: &str) -> (String, String) {
    let cleaned = topic.trim().trim_start_matches('#');
    let title_topic = if cleaned.len() <= 3 && cleaned.chars().all(|c| c.is_alphabetic()) {
        cleaned.to_uppercase()
    } else {
        // Capitalize first letter of each segment separated by / or -
        cleaned
            .split('/')
            .map(|part| {
                let mut c = part.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            })
            .collect::<Vec<_>>()
            .join(" - ")
    };

    let safe_filename = title_topic
        .replace('/', " - ")
        .replace('\\', " - ")
        .replace(':', " - ");

    let moc_title = format!("MOC - {}", safe_filename);
    let moc_filename = format!("{}.md", moc_title);
    (moc_title, moc_filename)
}

/// The core MOC and living index synthesis orchestrator.
pub struct MocSynthesizer {
    vault_root: PathBuf,
}

impl MocSynthesizer {
    /// Creates a new MocSynthesizer for the given vault root directory.
    pub fn new<P: AsRef<Path>>(vault_root: P) -> Self {
        Self {
            vault_root: vault_root.as_ref().to_path_buf(),
        }
    }

    /// Discovers all markdown documents in the vault, parses them, and builds a KnowledgeGraph.
    pub fn load_vault_documents(&self) -> Result<(Vec<ParsedDocument>, KnowledgeGraph)> {
        let parser = MarkdownParser::new();
        let mut docs = Vec::new();

        let walker = WalkBuilder::new(&self.vault_root)
            .hidden(true)
            .parents(false)
            .git_ignore(true)
            .build();

        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("md") {
                continue;
            }

            let rel_path = match path.strip_prefix(&self.vault_root) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let rel_str = rel_path.to_string_lossy();
            if rel_str.starts_with(".atlaswiki")
                || rel_str.starts_with(".git")
                || rel_str.starts_with(".obsidian")
                || rel_str.contains("node_modules")
            {
                continue;
            }

            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(doc) = parser.parse_file(rel_path, &content) {
                    docs.push(doc);
                }
            }
        }

        let mut kg = KnowledgeGraph::from_documents(&docs);
        kg.compute_pagerank(0.85, 50);

        Ok((docs, kg))
    }

    /// Converts parsed documents into `NoteSummary` records with PageRank and degree metrics.
    pub fn extract_note_summaries(
        &self,
        docs: &[ParsedDocument],
        kg: &KnowledgeGraph,
    ) -> Vec<NoteSummary> {
        let mut summaries = Vec::new();

        for doc in docs {
            let tags: Vec<String> = doc.tags.iter().map(|t| t.name.clone()).collect();
            let (in_deg, out_deg) = kg.degree(&doc.title).unwrap_or((0, 0));
            let pr = kg.get_pagerank(&doc.title);

            summaries.push(NoteSummary {
                title: doc.title.clone(),
                path: doc.path.to_string_lossy().to_string(),
                tags,
                word_count: doc.word_count,
                pagerank: pr,
                in_degree: in_deg,
                out_degree: out_deg,
            });
        }

        summaries.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.title.cmp(&b.title))
        });

        summaries
    }

    /// Discovers all unique top-level topics (tags) across the notes.
    pub fn discover_topics(&self, notes: &[NoteSummary]) -> Vec<String> {
        let mut topics = BTreeSet::new();

        for note in notes {
            for tag in &note.tags {
                let clean = tag.trim().trim_start_matches('#');
                if clean.is_empty() {
                    continue;
                }
                let top_segment = clean.split('/').next().unwrap_or(clean);
                topics.insert(top_segment.to_string());
            }
        }

        topics.into_iter().collect()
    }

    /// Synthesizes the markdown body for a specific topic MOC.
    pub fn synthesize_topic_moc(
        &self,
        topic: &str,
        all_notes: &[NoteSummary],
        all_docs: &[ParsedDocument],
    ) -> (String, Vec<String>) {
        let clean_topic = topic.trim().trim_start_matches('#').to_lowercase();

        // Filter notes belonging to this topic
        let mut topic_notes: Vec<NoteSummary> = all_notes
            .iter()
            .filter(|n| {
                // Check tags
                let tag_match = n.tags.iter().any(|t| {
                    let c = t.trim().trim_start_matches('#').to_lowercase();
                    c == clean_topic
                        || c.starts_with(&format!("{}/", clean_topic))
                        || c.split('/').any(|seg| seg == clean_topic)
                });
                if tag_match {
                    return true;
                }
                // Check title
                n.title.to_lowercase().contains(&clean_topic)
            })
            .cloned()
            .collect();

        // Sort by PageRank descending
        topic_notes.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.title.cmp(&b.title))
        });

        let mut out = String::new();
        let (_moc_title, _) = sanitize_topic_name(topic);

        out.push_str(&format!("## Topic Overview\n\n"));
        out.push_str(&format!("- **Topic**: `{}`\n", topic));
        out.push_str(&format!("- **Total Notes**: {}\n", topic_notes.len()));
        out.push_str("- **Ordering**: Hub Centrality (PageRank descending)\n\n");

        // Key Hubs
        let hub_count = topic_notes.len().min(5);
        let hub_notes: Vec<String> = topic_notes.iter().take(hub_count).map(|n| n.title.clone()).collect();

        if !hub_notes.is_empty() {
            out.push_str("### Key Hub Notes & Core Concepts\n\n");
            for note in topic_notes.iter().take(hub_count) {
                out.push_str(&format!(
                    "- [[{}]] · Hub Score: {:.2} (In: {}, Out: {})\n",
                    note.title, note.pagerank, note.in_degree, note.out_degree
                ));
            }
            out.push('\n');
        }

        // Tag Taxonomy Tree
        let mut taxonomy = TagTaxonomy::default();
        for note in &topic_notes {
            for tag in &note.tags {
                taxonomy.insert(tag, note);
            }
        }
        taxonomy.sort_by_pagerank();

        if !taxonomy.children.is_empty() {
            out.push_str("### Taxonomy Breakdown\n\n");
            taxonomy.render_markdown(1, &mut out);
        }

        // All Notes Listing
        out.push_str("### Note Index\n\n");
        if topic_notes.is_empty() {
            out.push_str("_No notes currently categorized under this topic._\n\n");
        } else {
            for note in &topic_notes {
                out.push_str(&format!(
                    "- [[{}]] · Hub: {:.2}\n",
                    note.title, note.pagerank
                ));
            }
            out.push('\n');
        }

        // Related Outgoing Connections to other vault notes
        let topic_title_set: BTreeSet<String> = topic_notes.iter().map(|n| n.title.to_lowercase()).collect();
        let mut outgoing_connections: BTreeMap<String, usize> = BTreeMap::new();

        for doc in all_docs {
            if topic_title_set.contains(&doc.title.to_lowercase()) {
                for link in &doc.links {
                    let target_lower = link.target_note.to_lowercase();
                    if !topic_title_set.contains(&target_lower) && !target_lower.is_empty() {
                        *outgoing_connections.entry(link.target_note.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        if !outgoing_connections.is_empty() {
            out.push_str("### Connected Vault Concepts\n\n");
            out.push_str("Outgoing references connecting this topic to broader knowledge:\n\n");
            let mut sorted_conn: Vec<(String, usize)> = outgoing_connections.into_iter().collect();
            sorted_conn.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (target, count) in sorted_conn.iter().take(10) {
                out.push_str(&format!("- [[{}]] ({} references)\n", target, count));
            }
            out.push('\n');
        }

        (out, hub_notes)
    }

    /// Synthesizes the living wiki index (`Index.md`).
    pub fn synthesize_living_index(
        &self,
        notes: &[NoteSummary],
        mocs: &[TopicMocInfo],
        orphans: &[NoteSummary],
        wanted: &[(String, usize)],
    ) -> String {
        let mut out = String::new();

        // 1. Vault Overview
        let total_words: usize = notes.iter().map(|n| n.word_count).sum();
        out.push_str("## Vault Overview\n\n");
        out.push_str(&format!("- **Total Notes**: {}\n", notes.len()));
        out.push_str(&format!("- **Topic MOCs**: {}\n", mocs.len()));
        out.push_str(&format!("- **Total Words**: {}\n", total_words));
        out.push_str(&format!("- **Orphan Notes**: {} ([[Orphans]])\n", orphans.len()));
        out.push_str(&format!("- **Wanted Notes**: {}\n\n", wanted.len()));

        // 2. Maps of Content (MOCs)
        if !mocs.is_empty() {
            out.push_str("## Maps of Content (MOCs)\n\n");
            for moc in mocs {
                out.push_str(&format!(
                    "- [[{}]] ({} notes)\n",
                    moc.title, moc.note_count
                ));
            }
            out.push('\n');
        }

        // 3. Key Vault Hubs (PageRank Centrality)
        let hub_count = notes.len().min(7);
        if hub_count > 0 {
            out.push_str("## Key Vault Hubs (High Centrality)\n\n");
            for note in notes.iter().take(hub_count) {
                out.push_str(&format!(
                    "- [[{}]] · PageRank: {:.2} (In: {}, Out: {})\n",
                    note.title, note.pagerank, note.in_degree, note.out_degree
                ));
            }
            out.push('\n');
        }

        // 4. Tag Taxonomy Tree
        let mut taxonomy = TagTaxonomy::default();
        let mut untagged = Vec::new();

        for note in notes {
            if note.tags.is_empty() {
                untagged.push(note.clone());
            } else {
                for tag in &note.tags {
                    taxonomy.insert(tag, note);
                }
            }
        }
        taxonomy.sort_by_pagerank();

        if !taxonomy.children.is_empty() {
            out.push_str("## Vault Taxonomy\n\n");
            taxonomy.render_markdown(1, &mut out);
        }

        // 5. Untagged Notes
        if !untagged.is_empty() {
            out.push_str("## Untagged Notes\n\n");
            untagged.sort_by(|a, b| {
                b.pagerank
                    .partial_cmp(&a.pagerank)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.title.cmp(&b.title))
            });
            for note in &untagged {
                out.push_str(&format!("- [[{}]] · Hub: {:.2}\n", note.title, note.pagerank));
            }
            out.push('\n');
        }

        // 6. Vault Health & Maintenance
        out.push_str("## Vault Health\n\n");
        if orphans.is_empty() {
            out.push_str("- ✓ **Orphans**: Zero isolated notes detected. Vault graph is connected.\n");
        } else {
            out.push_str(&format!(
                "- ⚡ **Orphans**: [[Orphans]] ({} notes need connections)\n",
                orphans.len()
            ));
        }

        if wanted.is_empty() {
            out.push_str("- ✓ **Wanted Pages**: All wikilinks resolve to existing notes.\n");
        } else {
            out.push_str(&format!(
                "- ⚡ **Wanted Pages**: {} unresolved references.\n",
                wanted.len()
            ));
        }

        out
    }

    /// Synthesizes the isolated notes catalog (`Orphans.md`).
    pub fn synthesize_orphans_index(&self, orphans: &[NoteSummary]) -> String {
        let mut out = String::new();

        if orphans.is_empty() {
            out.push_str("## Vault Connection Status: Healthy\n\n");
            out.push_str("✓ **Zero isolated orphan notes detected!** Every note in the vault has at least one incoming or outgoing wikilink.\n");
        } else {
            out.push_str(&format!("## Isolated Notes ({})\n\n", orphans.len()));
            out.push_str("The following notes have zero incoming and zero outgoing links in the knowledge graph:\n\n");

            for note in orphans {
                out.push_str(&format!(
                    "- [[{}]] (`{}`) · Words: {}\n",
                    note.title, note.path, note.word_count
                ));
            }

            out.push_str("\n### Suggested Next Steps\n\n");
            out.push_str("1. Connect these notes to relevant central hubs using `[[Note Title]]` wikilinks.\n");
            out.push_str("2. Reference them in appropriate Maps of Content (MOCs).\n");
            out.push_str("3. Add hierarchical tags (e.g. `#topic/subtopic`) so they integrate into the taxonomy tree.\n");
        }

        out
    }

    /// Generates or updates a Map of Content (MOC) for a specific topic or all topics.
    pub fn generate_moc(
        &self,
        topic: Option<&str>,
        dry_run: bool,
    ) -> Result<Vec<MocReport>> {
        let (docs, kg) = self.load_vault_documents()?;
        let notes = self.extract_note_summaries(&docs, &kg);

        let target_topics: Vec<String> = match topic {
            Some(t) => vec![t.to_string()],
            None => self.discover_topics(&notes),
        };

        if target_topics.is_empty() {
            let default_topic = "General".to_string();
            return self.generate_moc_single(&default_topic, &notes, &docs, dry_run).map(|r| vec![r]);
        }

        let mut reports = Vec::new();
        for top in &target_topics {
            let rep = self.generate_moc_single(top, &notes, &docs, dry_run)?;
            reports.push(rep);
        }

        Ok(reports)
    }

    fn generate_moc_single(
        &self,
        topic: &str,
        notes: &[NoteSummary],
        docs: &[ParsedDocument],
        dry_run: bool,
    ) -> Result<MocReport> {
        let (moc_title, moc_filename) = sanitize_topic_name(topic);
        let moc_path = self.vault_root.join(&moc_filename);

        let (body, hub_notes) = self.synthesize_topic_moc(topic, notes, docs);

        let default_header = format!(
            "# {}\n\n> Automated Map of Content synthesized by AtlasWiki for `{}`.\n> Edits outside synthesis markers are preserved.\n",
            moc_title, topic
        );

        let existing = if moc_path.is_file() {
            fs::read_to_string(&moc_path).unwrap_or_default()
        } else {
            String::new()
        };

        let updated = update_synthesis_content(&existing, &body, Some(&default_header));

        if !dry_run {
            fs::write(&moc_path, &updated)
                .with_context(|| format!("Failed to write MOC to {:?}", moc_path))?;
        }

        let clean_topic = topic.trim().trim_start_matches('#').to_lowercase();
        let note_count = notes
            .iter()
            .filter(|n| {
                n.tags.iter().any(|t| {
                    let c = t.trim().trim_start_matches('#').to_lowercase();
                    c == clean_topic || c.starts_with(&format!("{}/", clean_topic))
                }) || n.title.to_lowercase().contains(&clean_topic)
            })
            .count();

        Ok(MocReport {
            topic: topic.to_string(),
            file_path: moc_filename,
            note_count,
            hub_notes,
            dry_run,
            content: if dry_run { Some(updated) } else { None },
        })
    }

    /// Generates the living wiki index: `Index.md`, `Orphans.md`, and all topic MOCs.
    pub fn generate_living_index(&self, dry_run: bool) -> Result<LivingWikiReport> {
        let (docs, kg) = self.load_vault_documents()?;
        let notes = self.extract_note_summaries(&docs, &kg);
        let topics = self.discover_topics(&notes);

        // 1. Generate all topic MOCs
        let mut topic_mocs = Vec::new();
        for top in &topics {
            let moc_rep = self.generate_moc_single(top, &notes, &docs, dry_run)?;
            let (title, fname) = sanitize_topic_name(top);
            topic_mocs.push(TopicMocInfo {
                topic: top.clone(),
                file_name: fname,
                title,
                note_count: moc_rep.note_count,
                hub_notes: moc_rep.hub_notes,
            });
        }

        // 2. Identify orphans and wanted pages
        let orphan_nodes = kg.get_orphans();
        let orphan_titles: BTreeSet<String> = orphan_nodes.iter().map(|o| o.title.clone()).collect();
        let orphan_summaries: Vec<NoteSummary> = notes
            .iter()
            .filter(|n| orphan_titles.contains(&n.title))
            .cloned()
            .collect();

        let wanted = kg.get_wanted_pages();

        // 3. Synthesize and write Orphans.md
        let orphans_filename = "Orphans.md";
        let orphans_path = self.vault_root.join(orphans_filename);
        let orphans_body = self.synthesize_orphans_index(&orphan_summaries);
        let orphans_default_header = "# Orphan Notes Index\n\n> Catalog of notes with zero incoming and outgoing links.\n> Edits outside synthesis markers are preserved.\n";

        let existing_orphans = if orphans_path.is_file() {
            fs::read_to_string(&orphans_path).unwrap_or_default()
        } else {
            String::new()
        };

        let updated_orphans = update_synthesis_content(
            &existing_orphans,
            &orphans_body,
            Some(orphans_default_header),
        );

        if !dry_run {
            fs::write(&orphans_path, &updated_orphans)
                .with_context(|| format!("Failed to write {:?}", orphans_path))?;
        }

        // 4. Synthesize and write Index.md
        let index_filename = "Index.md";
        let index_path = self.vault_root.join(index_filename);
        let index_body = self.synthesize_living_index(&notes, &topic_mocs, &orphan_summaries, &wanted);
        let index_default_header = "# Vault Living Index\n\n> Automated living index synthesized by AtlasWiki.\n> Edits outside synthesis markers are preserved.\n";

        let existing_index = if index_path.is_file() {
            fs::read_to_string(&index_path).unwrap_or_default()
        } else {
            String::new()
        };

        let updated_index = update_synthesis_content(
            &existing_index,
            &index_body,
            Some(index_default_header),
        );

        if !dry_run {
            fs::write(&index_path, &updated_index)
                .with_context(|| format!("Failed to write {:?}", index_path))?;
        }

        Ok(LivingWikiReport {
            index_path: index_filename.to_string(),
            orphans_path: orphans_filename.to_string(),
            topic_mocs,
            total_notes: notes.len(),
            total_orphans: orphan_summaries.len(),
            dry_run,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idempotent_replacement_preserves_user_edits() {
        let initial_user_file = r#"---
title: My Custom Index
---

# Welcome to My Vault!
This is my manually written introduction that should never be deleted.

<!-- ATLASWIKI:BEGIN_SYNTHESIS -->
Old generated content line 1
Old generated content line 2
<!-- ATLASWIKI:END_SYNTHESIS -->

## My Personal Scratchpad
- [ ] Read chapter 3
- [ ] Fix car
"#;

        let new_synthesis = "New synthesized MOC line 1\nNew synthesized MOC line 2";

        let updated = update_synthesis_content(initial_user_file, new_synthesis, None);

        // Verify user content preserved
        assert!(updated.contains("# Welcome to My Vault!"));
        assert!(updated.contains("This is my manually written introduction that should never be deleted."));
        assert!(updated.contains("## My Personal Scratchpad"));
        assert!(updated.contains("- [ ] Read chapter 3"));

        // Verify synthesis markers and new content present
        assert!(updated.contains(SYNTHESIS_BEGIN));
        assert!(updated.contains(SYNTHESIS_END));
        assert!(updated.contains("New synthesized MOC line 1"));
        assert!(!updated.contains("Old generated content line 1"));

        // Test idempotence: running again produces exact same output
        let updated_again = update_synthesis_content(&updated, new_synthesis, None);
        assert_eq!(updated, updated_again);
    }

    #[test]
    fn test_empty_content_with_default_header() {
        let default_header = "# MOC - AI\n\n> Tag taxonomy MOC";
        let body = "- [[Machine Learning]]\n- [[Deep Learning]]";

        let generated = update_synthesis_content("", body, Some(default_header));

        assert!(generated.starts_with("# MOC - AI"));
        assert!(generated.contains(SYNTHESIS_BEGIN));
        assert!(generated.contains("- [[Machine Learning]]"));
        assert!(generated.contains(SYNTHESIS_END));

        // Re-updating preserves header and replaces body
        let new_body = "- [[Machine Learning]]\n- [[Deep Learning]]\n- [[Transformers]]";
        let regenerated = update_synthesis_content(&generated, new_body, Some(default_header));
        assert!(regenerated.contains("- [[Transformers]]"));
        assert!(regenerated.starts_with("# MOC - AI"));
    }

    #[test]
    fn test_taxonomy_tree_insertion_and_ordering() {
        let mut taxonomy = TagTaxonomy::default();

        let n1 = NoteSummary {
            title: "Supervised Learning".to_string(),
            path: "ML/Supervised.md".to_string(),
            tags: vec!["ai/ml".to_string()],
            word_count: 500,
            pagerank: 0.8,
            in_degree: 5,
            out_degree: 2,
        };

        let n2 = NoteSummary {
            title: "Deep Learning".to_string(),
            path: "ML/Deep.md".to_string(),
            tags: vec!["ai/ml".to_string()],
            word_count: 800,
            pagerank: 0.95, // Higher pagerank should sort first
            in_degree: 10,
            out_degree: 4,
        };

        let n3 = NoteSummary {
            title: "Transformers".to_string(),
            path: "NLP/Transformers.md".to_string(),
            tags: vec!["ai/nlp".to_string()],
            word_count: 600,
            pagerank: 0.7,
            in_degree: 3,
            out_degree: 1,
        };

        taxonomy.insert("ai/ml", &n1);
        taxonomy.insert("ai/ml", &n2);
        taxonomy.insert("ai/nlp", &n3);
        taxonomy.sort_by_pagerank();

        let mut out = String::new();
        taxonomy.render_markdown(1, &mut out);

        // Verify hierarchy rendered
        assert!(out.contains("### #ai"));
        assert!(out.contains("#### #ai/ml"));
        assert!(out.contains("#### #ai/nlp"));

        // In ai/ml, Deep Learning (PR: 0.95) must appear before Supervised Learning (PR: 0.80)
        let pos_deep = out.find("[[Deep Learning]]").unwrap();
        let pos_supervised = out.find("[[Supervised Learning]]").unwrap();
        assert!(pos_deep < pos_supervised);
    }
}
