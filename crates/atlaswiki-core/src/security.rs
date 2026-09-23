//! Security module for AtlasWiki: path traversal sanitization, HTML escaping,
//! and DoS protection.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SecurityError {
    #[error("Path traversal detected: link target attempts to escape vault root ('..')")]
    ParentTraversalAttempt,
    #[error("Absolute path escape detected: '{0}'")]
    AbsolutePathEscape(String),
    #[error("Path contains forbidden characters or null bytes")]
    InvalidCharacters,
    #[error("Symlink target escapes vault root: '{target}' is outside vault")]
    SymlinkEscapesRoot { target: PathBuf },
    #[error("File size {size} bytes exceeds maximum allowed limit of {max_size} bytes")]
    FileTooLarge { size: u64, max_size: u64 },
    #[error("Frontmatter exceeds maximum size limit of {0} bytes")]
    FrontmatterTooLarge(usize),
    #[error("YAML alias/anchor bomb detected: found {0} aliases/anchors")]
    YamlBombDetected(usize),
    #[error("Embed recursion depth exceeded (max {0})")]
    EmbedDepthExceeded(usize),
    #[error("Cyclic embed detected for note '{0}'")]
    CyclicEmbedDetected(String),
    #[error("IO error: {0}")]
    IoError(String),
}

/// VaultRoot enforces boundaries so markdown links never escape the vault directory.
#[derive(Debug, Clone)]
pub struct VaultRoot {
    canonical_root: PathBuf,
}

impl VaultRoot {
    pub fn new<P: AsRef<Path>>(root_path: P) -> Result<Self, SecurityError> {
        let canonical_root = std::fs::canonicalize(root_path.as_ref())
            .map_err(|e| SecurityError::IoError(e.to_string()))?;
        Ok(Self { canonical_root })
    }

    pub fn canonical_path(&self) -> &Path {
        &self.canonical_root
    }

    /// Pure lexical path sanitization.
    /// Resolves `.` and `..` without touching disk. Rejects attempts to escape root.
    pub fn lexical_sanitize_relative(&self, raw_path: &str) -> Result<PathBuf, SecurityError> {
        if raw_path.contains('\0') {
            return Err(SecurityError::InvalidCharacters);
        }

        let normalized = raw_path.replace('\\', "/");
        if normalized.starts_with('/')
            || (normalized.len() >= 2
                && normalized.as_bytes()[1] == b':'
                && normalized.as_bytes()[0].is_ascii_alphabetic())
        {
            return Err(SecurityError::AbsolutePathEscape(raw_path.to_string()));
        }
        let path = Path::new(&normalized);

        let mut components_stack = Vec::new();

        for comp in path.components() {
            match comp {
                Component::Prefix(p) => {
                    return Err(SecurityError::AbsolutePathEscape(format!("{:?}", p)));
                }
                Component::RootDir => {
                    return Err(SecurityError::AbsolutePathEscape(raw_path.to_string()));
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    if components_stack.pop().is_none() {
                        return Err(SecurityError::ParentTraversalAttempt);
                    }
                }
                Component::Normal(c) => {
                    components_stack.push(c);
                }
            }
        }

        let mut sanitized = PathBuf::new();
        for comp in components_stack {
            sanitized.push(comp);
        }

        Ok(sanitized)
    }

    /// Resolves a target path relative to base directory while verifying it stays inside the vault.
    pub fn resolve_within_vault(
        &self,
        base_rel_dir: &Path,
        untrusted_target: &str,
    ) -> Result<PathBuf, SecurityError> {
        let sanitized_rel = self.lexical_sanitize_relative(untrusted_target)?;
        let candidate_full = self.canonical_root.join(base_rel_dir).join(&sanitized_rel);

        if candidate_full.exists() {
            let canonical_target = candidate_full
                .canonicalize()
                .map_err(|e| SecurityError::IoError(e.to_string()))?;

            canonical_target
                .strip_prefix(&self.canonical_root)
                .map_err(|_| SecurityError::SymlinkEscapesRoot {
                    target: canonical_target.clone(),
                })?;

            Ok(canonical_target)
        } else {
            let combined = self.canonical_root.join(base_rel_dir).join(&sanitized_rel);
            let mut stack = Vec::new();

            for comp in combined.components() {
                match comp {
                    Component::ParentDir => {
                        stack.pop();
                    }
                    Component::CurDir => {}
                    other => stack.push(other.as_os_str()),
                }
            }

            let mut result = PathBuf::new();
            for part in stack {
                result.push(part);
            }

            result
                .strip_prefix(&self.canonical_root)
                .map_err(|_| SecurityError::ParentTraversalAttempt)?;

            Ok(result)
        }
    }

    /// Safely reads file content inside the vault root, enforcing boundary containment,
    /// symlink jail verification, and DoS file size limits.
    pub fn safe_read_file<P: AsRef<Path>>(&self, untrusted_rel_path: P) -> Result<String, SecurityError> {
        let untrusted_str = untrusted_rel_path.as_ref().to_string_lossy();
        let canonical_target = self.resolve_within_vault(Path::new(""), &untrusted_str)?;

        let meta = std::fs::metadata(&canonical_target)
            .map_err(|e| SecurityError::IoError(e.to_string()))?;

        if meta.len() > 10 * 1024 * 1024 {
            return Err(SecurityError::FileTooLarge {
                size: meta.len(),
                max_size: 10 * 1024 * 1024,
            });
        }

        std::fs::read_to_string(&canonical_target)
            .map_err(|e| SecurityError::IoError(e.to_string()))
    }
}

/// HTML escaping and link sanitization for safe rendering.
pub struct MarkdownSanitizer;

impl MarkdownSanitizer {
    pub fn escape_html(input: &str) -> String {
        let mut output = String::with_capacity(input.len());
        for c in input.chars() {
            match c {
                '&' => output.push_str("&amp;"),
                '<' => output.push_str("&lt;"),
                '>' => output.push_str("&gt;"),
                '"' => output.push_str("&quot;"),
                '\'' => output.push_str("&#x27;"),
                '/' => output.push_str("&#x2F;"),
                _ => output.push(c),
            }
        }
        output
    }

    pub fn is_safe_url(url: &str) -> bool {
        let filtered: String = url
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_ascii_control())
            .collect();
        let lower = filtered.to_ascii_lowercase();

        if lower.starts_with("javascript:")
            || lower.starts_with("vbscript:")
            || lower.starts_with("file:")
            || lower.starts_with("data:")
        {
            return false;
        }

        lower.starts_with("http://")
            || lower.starts_with("https://")
            || lower.starts_with("mailto://")
            || lower.starts_with('#')
            || !lower.contains(':')
    }

    pub fn safe_json_for_script<T: serde::Serialize>(data: &T) -> Result<String, serde_json::Error> {
        let json_str = serde_json::to_string(data)?;
        let escaped = json_str
            .replace('<', "\\u003c")
            .replace('>', "\\u003e")
            .replace('&', "\\u0026");
        Ok(escaped)
    }
}

/// Denial of Service guards for file size and recursive embedding.
pub struct DosGuards {
    pub max_note_size: u64,
    pub max_embed_depth: usize,
}

impl Default for DosGuards {
    fn default() -> Self {
        Self {
            max_note_size: 10 * 1024 * 1024, // 10 MB limit
            max_embed_depth: 5,
        }
    }
}

impl DosGuards {
    pub fn check_file_size(&self, path: &Path) -> Result<(), SecurityError> {
        let meta = std::fs::metadata(path).map_err(|e| SecurityError::IoError(e.to_string()))?;
        if meta.len() > self.max_note_size {
            return Err(SecurityError::FileTooLarge {
                size: meta.len(),
                max_size: self.max_note_size,
            });
        }
        Ok(())
    }

    pub fn expand_embeds_safe<F>(
        &self,
        target_note: &str,
        visited: &mut HashSet<String>,
        current_depth: usize,
        content_loader: &F,
    ) -> Result<String, SecurityError>
    where
        F: Fn(&str) -> Option<String>,
    {
        if current_depth > self.max_embed_depth {
            return Ok(format!("<!-- [Embed depth limit reached for '{}'] -->", target_note));
        }

        if !visited.insert(target_note.to_string()) {
            return Ok(format!("<!-- [Cyclic embed detected for '{}'] -->", target_note));
        }

        let raw_content = content_loader(target_note).unwrap_or_default();
        visited.remove(target_note);
        Ok(raw_content)
    }
}
