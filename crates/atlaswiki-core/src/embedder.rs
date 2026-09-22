//! Pure-Rust Model2Vec static vector embedding engine for AtlasWiki.
//! Zero C++ dependencies, safetensors model weights, pure-Rust tokenizer.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use half::f16;
use rayon::prelude::*;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use tokenizers::Tokenizer;

/// Model configuration parsed from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct Model2VecConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub hidden_dim: Option<usize>,
    #[serde(default)]
    pub apply_pca: Option<usize>,
    #[serde(default)]
    pub apply_zipf: Option<bool>,
    #[serde(default = "default_normalize")]
    pub normalize: bool,
    #[serde(default = "default_seq_length")]
    pub seq_length: usize,
}

fn default_model_type() -> String {
    "model2vec".to_string()
}
fn default_normalize() -> bool {
    true
}
fn default_seq_length() -> usize {
    512
}

/// Pure-Rust Model2Vec Embedder.
/// Immutable internal representation allows lock-free concurrent access across Rayon threads.
#[derive(Debug, Clone)]
pub struct Model2VecEmbedder {
    inner: Arc<EmbedderInner>,
}

#[derive(Debug)]
struct EmbedderInner {
    tokenizer: Tokenizer,
    embeddings: Vec<f32>,
    weights: Option<Vec<f32>>,
    token_mapping: Option<Vec<usize>>,
    dim: usize,
    vocab_size: usize,
    normalize: bool,
    max_seq_length: usize,
}

impl Model2VecEmbedder {
    /// Load Model2Vec from a local directory containing tokenizer.json, model.safetensors, and config.json
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        let model_path = dir.join("model.safetensors");
        let config_path = dir.join("config.json");

        if !tokenizer_path.exists() || !model_path.exists() || !config_path.exists() {
            return Err(anyhow!(
                "Missing model files in {:?}. Required: tokenizer.json, model.safetensors, config.json",
                dir
            ));
        }

        let tokenizer_bytes = fs::read(&tokenizer_path)
            .with_context(|| format!("Failed to read {:?}", tokenizer_path))?;
        let model_bytes = fs::read(&model_path)
            .with_context(|| format!("Failed to read {:?}", model_path))?;
        let config_bytes = fs::read(&config_path)
            .with_context(|| format!("Failed to read {:?}", config_path))?;

        Self::from_bytes(&tokenizer_bytes, &model_bytes, &config_bytes)
    }

    /// Load Model2Vec directly from in-memory byte slices.
    pub fn from_bytes(
        tokenizer_bytes: &[u8],
        model_bytes: &[u8],
        config_bytes: &[u8],
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow!("Failed to deserialize tokenizer: {e}"))?;

        let config: Model2VecConfig = serde_json::from_slice(config_bytes)
            .context("Failed to parse Model2Vec config.json")?;

        let safetensors = SafeTensors::deserialize(model_bytes)
            .context("Failed to deserialize model.safetensors")?;

        let tensor = safetensors
            .tensor("embeddings")
            .or_else(|_| safetensors.tensor("embeddings.weight"))
            .or_else(|_| safetensors.tensor("0"))
            .context("Embedding tensor not found in safetensors")?;

        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(anyhow!("Embedding tensor must be 2D, got shape: {:?}", shape));
        }
        let vocab_size = shape[0];
        let dim = shape[1];

        let raw_data = tensor.data();
        let embeddings: Vec<f32> = match tensor.dtype() {
            Dtype::F32 => raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F16 => raw_data
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            Dtype::I8 => raw_data.iter().map(|&b| f32::from(b as i8)).collect(),
            other => return Err(anyhow!("Unsupported tensor dtype: {:?}", other)),
        };

        if embeddings.len() != vocab_size * dim {
            return Err(anyhow!(
                "Decoded embeddings length {} != vocab_size {} * dim {}",
                embeddings.len(),
                vocab_size,
                dim
            ));
        }

        let weights = if let Ok(wt) = safetensors.tensor("weights") {
            let raw = wt.data();
            let w_vec: Vec<f32> = match wt.dtype() {
                Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => return Err(anyhow!("Unsupported weights dtype")),
            };
            Some(w_vec)
        } else {
            None
        };

        let token_mapping = if let Ok(mt) = safetensors.tensor("mapping") {
            let raw = mt.data();
            let m_vec: Vec<usize> = match mt.dtype() {
                Dtype::I64 => raw
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as usize)
                    .collect(),
                Dtype::I32 => raw
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize)
                    .collect(),
                _ => return Err(anyhow!("Unsupported mapping dtype")),
            };
            Some(m_vec)
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(EmbedderInner {
                tokenizer,
                embeddings,
                weights,
                token_mapping,
                dim,
                vocab_size,
                normalize: config.normalize,
                max_seq_length: config.seq_length,
            }),
        })
    }

    /// Load or download model using Hugging Face Hub with local caching.
    pub fn from_pretrained(model_id: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_base = cache_dir.unwrap_or_else(|| {
            dirs_or_home()
                .join(".cache")
                .join("atlaswiki")
                .join("models")
        });
        let model_dir = cache_base.join(model_id.replace('/', "--"));

        if model_dir.join("tokenizer.json").exists()
            && model_dir.join("model.safetensors").exists()
            && model_dir.join("config.json").exists()
        {
            return Self::from_directory(&model_dir);
        }

        fs::create_dir_all(&model_dir)?;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("Failed to initialize Hugging Face API client")?;
        let repo = api.model(model_id.to_string());

        let tokenizer_src = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;
        let model_src = repo.get("model.safetensors").context("Failed to download model.safetensors")?;
        let config_src = repo.get("config.json").context("Failed to download config.json")?;

        fs::copy(tokenizer_src, model_dir.join("tokenizer.json"))?;
        fs::copy(model_src, model_dir.join("model.safetensors"))?;
        fs::copy(config_src, model_dir.join("config.json"))?;

        Self::from_directory(&model_dir)
    }

    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    /// Encode a single text string into an L2-normalized static vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .inner
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        let token_ids = encoding.get_ids();
        if token_ids.is_empty() {
            return Ok(vec![0.0; self.inner.dim]);
        }

        let mut vector = vec![0.0_f32; self.inner.dim];
        let mut total_weight = 0.0_f32;

        for &id in token_ids.iter().take(self.inner.max_seq_length) {
            let tok_idx = id as usize;
            let mapped_idx = self
                .inner
                .token_mapping
                .as_ref()
                .and_then(|m| m.get(tok_idx))
                .copied()
                .unwrap_or(tok_idx);

            if mapped_idx >= self.inner.vocab_size {
                continue;
            }

            let weight = self
                .inner
                .weights
                .as_ref()
                .and_then(|w| w.get(tok_idx))
                .copied()
                .unwrap_or(1.0);

            let row_offset = mapped_idx * self.inner.dim;
            let row = &self.inner.embeddings[row_offset..row_offset + self.inner.dim];

            for (acc, &val) in vector.iter_mut().zip(row.iter()) {
                *acc += val * weight;
            }
            total_weight += weight;
        }

        let denom = if total_weight > 0.0 { total_weight } else { 1.0 };
        for val in vector.iter_mut() {
            *val /= denom;
        }

        if self.inner.normalize {
            Self::l2_normalize(&mut vector);
        }

        Ok(vector)
    }

    /// Parallel batch encoding over multiple chunks using Rayon.
    pub fn embed_batch_parallel(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts
            .par_iter()
            .map(|text| self.embed_text(text))
            .collect()
    }

    /// Fast in-place L2 normalization.
    #[inline(always)]
    pub fn l2_normalize(vec: &mut [f32]) {
        let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
        let norm = sum_sq.sqrt().max(1e-12);
        let inv_norm = 1.0 / norm;
        for v in vec.iter_mut() {
            *v *= inv_norm;
        }
    }

    /// Fast cosine similarity between two L2-normalized vectors (dot product).
    #[inline(always)]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

/// Fallback helper to find home directory
fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Unified embedding engine handling Model2Vec or graceful offline fallback.
#[derive(Clone)]
pub enum EmbeddingEngine {
    Model2Vec(Arc<Model2VecEmbedder>),
    OfflineLexicalOnly,
}

impl EmbeddingEngine {
    pub fn init_with_fallback(model_id: &str, cache_dir: Option<PathBuf>) -> Self {
        match Model2VecEmbedder::from_pretrained(model_id, cache_dir) {
            Ok(embedder) => {
                EmbeddingEngine::Model2Vec(Arc::new(embedder))
            }
            Err(_) => {
                EmbeddingEngine::OfflineLexicalOnly
            }
        }
    }

    pub fn is_vector_enabled(&self) -> bool {
        matches!(self, EmbeddingEngine::Model2Vec(_))
    }

    pub fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        match self {
            EmbeddingEngine::Model2Vec(m) => m.embed_text(text).ok(),
            EmbeddingEngine::OfflineLexicalOnly => None,
        }
    }
}
