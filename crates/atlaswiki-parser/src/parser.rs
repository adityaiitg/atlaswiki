use std::path::Path;
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::ast::{AstChunk, AstLink, AstSection, AstTag, LinkType, ParsedDocument};
use crate::frontmatter::extract_frontmatter;

pub struct MarkdownParser {
    wikilink_re: Regex,
    tag_re: Regex,
    inline_code_re: Regex,
    heading_re: Regex,
    markdown_link_re: Regex,
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self {
            // Matches !?[[target#heading^block|alias]] or !?[[target#^block|alias]]
            wikilink_re: Regex::new(
                r"(?P<embed>!)?\[\[(?P<target>[^\]|#^]+)(?:#(?:(?:\^(?P<block>[^\]|]+))|(?P<heading>[^\]|^|]+)(?:\^(?P<block2>[^\]|]+))?))?(?:\^(?P<block_direct>[^\]|]+))?(?:\|(?P<alias>[^\]]+))?\]\]"
            ).unwrap(),
            // Matches #tag or #nested/tag, ensuring not preceded by word char or '#'
            tag_re: Regex::new(
                r"(?:^|[\s,;:({\[])#(?P<tag>[a-zA-Z][a-zA-Z0-9_\-\/]*)"
            ).unwrap(),
            // Inline code span `...`
            inline_code_re: Regex::new(r"`[^`]*`").unwrap(),
            // Markdown heading line ^(#{1,6})\s+(.*)$
            heading_re: Regex::new(r"^(?P<hashes>#{1,6})\s+(?P<title>.*)$").unwrap(),
            // Standard markdown link [text](url)
            markdown_link_re: Regex::new(r"\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)").unwrap(),
        }
    }

    pub fn parse_file(&self, path: &Path, content: &str) -> anyhow::Result<ParsedDocument> {
        let (frontmatter, body, line_offset) = extract_frontmatter(content);

        let default_title = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string());

        let lines: Vec<&str> = body.lines().collect();
        let total_lines = lines.len().max(1);

        // Content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let content_hash = format!("{:x}", hasher.finalize());

        // Word count
        let word_count = content.split_whitespace().count();

        // 1. First pass: scan headings and code block boundaries
        let mut in_code_block = false;
        let mut headings: Vec<(u8, String, usize)> = Vec::new(); // (level, heading_text, line_number)
        let mut code_block_mask: Vec<bool> = vec![false; lines.len()];

        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                in_code_block = !in_code_block;
                code_block_mask[idx] = true;
                continue;
            }
            if in_code_block {
                code_block_mask[idx] = true;
                continue;
            }

            // Check if heading
            if let Some(cap) = self.heading_re.captures(trimmed) {
                let lvl = cap.name("hashes").unwrap().as_str().len() as u8;
                let title = cap.name("title").unwrap().as_str().trim().to_string();
                let line_num = line_offset + idx + 1;
                headings.push((lvl, title, line_num));
            }
        }

        // 2. Build Hierarchical Sections
        let mut sections: Vec<AstSection> = Vec::new();
        let mut heading_stack: Vec<(u8, String, String)> = Vec::new(); // (level, id, heading)
        let mut first_h1_title: Option<String> = None;

        if headings.is_empty() {
            // Document has no headings: create root overview section
            sections.push(AstSection {
                id: format!("{}#overview", slugify(&default_title)),
                heading: default_title.clone(),
                level: 1,
                parent_id: None,
                breadcrumbs: vec![default_title.clone()],
                line_start: line_offset + 1,
                line_end: line_offset + total_lines,
                content: body.trim().to_string(),
            });
        } else {
            for (idx, &(lvl, ref h_text, h_line)) in headings.iter().enumerate() {
                if lvl == 1 && first_h1_title.is_none() {
                    first_h1_title = Some(h_text.clone());
                }

                // Pop headings with stack_lvl >= current lvl
                while let Some(&(stack_lvl, _, _)) = heading_stack.last() {
                    if stack_lvl >= lvl {
                        heading_stack.pop();
                    } else {
                        break;
                    }
                }

                let slug = slugify(h_text);
                let section_id = format!("{}#{}", slugify(&default_title), slug);
                let parent_id = heading_stack.last().map(|(_, id, _)| id.clone());

                let mut breadcrumbs: Vec<String> =
                    heading_stack.iter().map(|(_, _, h)| h.clone()).collect();
                breadcrumbs.push(h_text.clone());

                heading_stack.push((lvl, section_id.clone(), h_text.clone()));

                // Determine end line: next heading line - 1 or end of doc
                let line_end = if idx + 1 < headings.len() {
                    headings[idx + 1].2.saturating_sub(1)
                } else {
                    line_offset + total_lines
                };

                let local_start = h_line.saturating_sub(line_offset + 1);
                let local_end = (line_end.saturating_sub(line_offset)).min(lines.len());
                let sec_content = if local_start < lines.len() && local_start < local_end {
                    lines[local_start..local_end].join("\n")
                } else {
                    String::new()
                };

                sections.push(AstSection {
                    id: section_id,
                    heading: h_text.clone(),
                    level: lvl,
                    parent_id,
                    breadcrumbs,
                    line_start: h_line,
                    line_end,
                    content: sec_content,
                });
            }
        }

        // Determine effective document title
        let final_title = frontmatter
            .title
            .clone()
            .or(first_h1_title)
            .unwrap_or(default_title);

        // 3. Extract Links and Tags (excluding code blocks and inline code)
        let mut links: Vec<AstLink> = Vec::new();
        let mut tags: Vec<AstTag> = Vec::new();

        // Frontmatter tags
        for fmt_tag in &frontmatter.tags {
            tags.push(AstTag {
                name: fmt_tag.clone(),
                line_number: 1,
                section_id: None,
            });
        }

        for (idx, line) in lines.iter().enumerate() {
            if code_block_mask[idx] {
                continue;
            }

            let line_num = line_offset + idx + 1;
            let current_section_id = sections
                .iter()
                .find(|s| line_num >= s.line_start && line_num <= s.line_end)
                .map(|s| s.id.clone());

            // A. Wikilinks & Embeds (raw line to preserve syntax)
            for cap in self.wikilink_re.captures_iter(line) {
                let is_embed = cap.name("embed").is_some();
                let target_raw = cap.name("target").map(|m| m.as_str().trim()).unwrap_or("");
                let heading_target = cap.name("heading").map(|m| m.as_str().trim().to_string());
                let block_target = cap
                    .name("block")
                    .or_else(|| cap.name("block2"))
                    .or_else(|| cap.name("block_direct"))
                    .map(|m| m.as_str().trim().to_string());
                let alias = cap.name("alias").map(|m| m.as_str().trim().to_string());

                if !target_raw.is_empty() {
                    let snippet = line.trim().to_string();

                    links.push(AstLink {
                        link_type: if is_embed {
                            LinkType::Embed
                        } else {
                            LinkType::Wikilink
                        },
                        target_note: target_raw.to_string(),
                        target_heading: heading_target,
                        target_block: block_target,
                        alias,
                        line_number: line_num,
                        context_snippet: Some(snippet),
                        source_section_id: current_section_id.clone(),
                    });
                }
            }

            // B. Standard Markdown Links
            for cap in self.markdown_link_re.captures_iter(line) {
                let text = cap.name("text").map(|m| m.as_str().trim().to_string());
                let url = cap.name("url").map(|m| m.as_str().trim()).unwrap_or("");
                let is_internal_md = url.ends_with(".md")
                    || (!url.starts_with("http://")
                        && !url.starts_with("https://")
                        && !url.starts_with('#'));

                if is_internal_md && !url.is_empty() {
                    let target_clean = url.trim_end_matches(".md").to_string();
                    links.push(AstLink {
                        link_type: LinkType::Markdown,
                        target_note: target_clean,
                        target_heading: None,
                        target_block: None,
                        alias: text,
                        line_number: line_num,
                        context_snippet: Some(line.trim().to_string()),
                        source_section_id: current_section_id.clone(),
                    });
                }
            }

            // C. Tags (strip inline code first so `def #foo` is ignored)
            let sanitized_line = self.inline_code_re.replace_all(line, " ");
            for cap in self.tag_re.captures_iter(&sanitized_line) {
                if let Some(tag_match) = cap.name("tag") {
                    let tag_str = tag_match.as_str().trim();
                    if !tag_str.is_empty()
                        && !tags.iter().any(|t| t.name == tag_str && t.line_number == line_num)
                    {
                        tags.push(AstTag {
                            name: tag_str.to_string(),
                            line_number: line_num,
                            section_id: current_section_id.clone(),
                        });
                    }
                }
            }
        }

        // 4. Generate Semantic Chunks
        let chunks = self.generate_chunks(&final_title, &sections, body, line_offset);

        Ok(ParsedDocument {
            path: path.to_path_buf(),
            title: final_title,
            frontmatter,
            sections,
            links,
            tags,
            chunks,
            word_count,
            content_hash,
        })
    }

    fn generate_chunks(
        &self,
        doc_title: &str,
        sections: &[AstSection],
        body: &str,
        line_offset: usize,
    ) -> Vec<AstChunk> {
        let mut chunks = Vec::new();
        let doc_slug = slugify(doc_title);

        for (sec_idx, sec) in sections.iter().enumerate() {
            let breadcrumbs_str = sec.breadcrumbs.join(" > ");
            let content = sec.content.trim();

            if content.is_empty() {
                continue;
            }

            let words: Vec<&str> = content.split_whitespace().collect();
            if words.len() <= 400 {
                chunks.push(AstChunk {
                    chunk_id: format!("{doc_slug}::{sec_idx}::0"),
                    section_id: Some(sec.id.clone()),
                    title: doc_title.to_string(),
                    breadcrumbs: breadcrumbs_str,
                    line_start: sec.line_start,
                    line_end: sec.line_end,
                    content: content.to_string(),
                });
            } else {
                let paragraphs = content.split("\n\n");
                let mut current_chunk_words = 0;
                let mut current_chunk_text = String::new();
                let mut chunk_sub_idx = 0;

                for para in paragraphs {
                    let para_trimmed = para.trim();
                    if para_trimmed.is_empty() {
                        continue;
                    }
                    let p_words = para_trimmed.split_whitespace().count();

                    if current_chunk_words + p_words > 300 && !current_chunk_text.is_empty() {
                        chunks.push(AstChunk {
                            chunk_id: format!("{doc_slug}::{sec_idx}::{chunk_sub_idx}"),
                            section_id: Some(sec.id.clone()),
                            title: doc_title.to_string(),
                            breadcrumbs: breadcrumbs_str.clone(),
                            line_start: sec.line_start,
                            line_end: sec.line_end,
                            content: current_chunk_text.trim().to_string(),
                        });
                        chunk_sub_idx += 1;
                        current_chunk_text.clear();
                        current_chunk_words = 0;
                    }

                    if !current_chunk_text.is_empty() {
                        current_chunk_text.push_str("\n\n");
                    }
                    current_chunk_text.push_str(para_trimmed);
                    current_chunk_words += p_words;
                }

                if !current_chunk_text.is_empty() {
                    chunks.push(AstChunk {
                        chunk_id: format!("{doc_slug}::{sec_idx}::{chunk_sub_idx}"),
                        section_id: Some(sec.id.clone()),
                        title: doc_title.to_string(),
                        breadcrumbs: breadcrumbs_str,
                        line_start: sec.line_start,
                        line_end: sec.line_end,
                        content: current_chunk_text.trim().to_string(),
                    });
                }
            }
        }

        if chunks.is_empty() && !body.trim().is_empty() {
            chunks.push(AstChunk {
                chunk_id: format!("{doc_slug}::0::0"),
                section_id: None,
                title: doc_title.to_string(),
                breadcrumbs: doc_title.to_string(),
                line_start: line_offset + 1,
                line_end: line_offset + body.lines().count(),
                content: body.trim().to_string(),
            });
        }

        chunks
    }
}

pub fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}
