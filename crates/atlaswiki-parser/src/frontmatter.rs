use serde_yaml::Value as YamlValue;
use crate::ast::Frontmatter;

pub fn extract_frontmatter(content: &str) -> (Frontmatter, &str, usize) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (Frontmatter::default(), content, 0);
    }

    // Find the end of the frontmatter block
    let after_first = &trimmed[3..];
    if let Some(first_newline) = after_first.find('\n') {
        let rest = &after_first[first_newline + 1..];
        if let Some(end_idx) = rest.find("\n---") {
            let yaml_str = &rest[..end_idx];
            let after_closing = &rest[end_idx + 4..];
            let body_start = if let Some(body_idx) = after_closing.find('\n') {
                &after_closing[body_idx + 1..]
            } else {
                after_closing
            };

            // Calculate number of lines consumed by frontmatter
            let frontmatter_chars = content.len() - body_start.len();
            let line_offset = content[..frontmatter_chars].lines().count();

            if let Ok(parsed_yaml) = serde_yaml::from_str::<YamlValue>(yaml_str) {
                let frontmatter = parse_yaml_map(parsed_yaml);
                return (frontmatter, body_start, line_offset);
            }
        }
    }

    (Frontmatter::default(), content, 0)
}

fn parse_yaml_map(yaml: YamlValue) -> Frontmatter {
    let mut fm = Frontmatter::default();
    if let YamlValue::Mapping(map) = yaml {
        for (k, v) in map {
            if let YamlValue::String(key) = k {
                match key.to_lowercase().as_str() {
                    "title" => {
                        if let Some(s) = yaml_value_to_string(&v) {
                            fm.title = Some(s);
                        }
                    }
                    "aliases" | "alias" => {
                        fm.aliases = yaml_value_to_string_list(&v);
                    }
                    "tags" | "tag" => {
                        fm.tags = yaml_value_to_string_list(&v);
                    }
                    "created" | "date" => {
                        if let Some(s) = yaml_value_to_string(&v) {
                            fm.created = Some(s);
                        }
                    }
                    "updated" | "modified" | "last_modified" => {
                        if let Some(s) = yaml_value_to_string(&v) {
                            fm.updated = Some(s);
                        }
                    }
                    _ => {
                        if let Ok(json_val) = serde_json::to_value(&v) {
                            fm.extra.insert(key, json_val);
                        }
                    }
                }
            }
        }
    }
    fm
}

fn yaml_value_to_string(val: &YamlValue) -> Option<String> {
    match val {
        YamlValue::String(s) => Some(s.clone()),
        YamlValue::Number(n) => Some(n.to_string()),
        YamlValue::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn yaml_value_to_string_list(val: &YamlValue) -> Vec<String> {
    let mut results = Vec::new();
    match val {
        YamlValue::Sequence(seq) => {
            for item in seq {
                if let Some(s) = yaml_value_to_string(item) {
                    let cleaned = s.trim().trim_start_matches('#').to_string();
                    if !cleaned.is_empty() {
                        results.push(cleaned);
                    }
                }
            }
        }
        YamlValue::String(s) => {
            // Could be comma or space separated: "tag1, tag2"
            for part in s.split(&[',', ' '][..]) {
                let cleaned = part.trim().trim_start_matches('#').to_string();
                if !cleaned.is_empty() {
                    results.push(cleaned);
                }
            }
        }
        _ => {}
    }
    results
}
