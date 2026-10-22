//! One encoder interface over the two embedding backends: the local ONNX
//! `Embedder` (the default) and, with the `remote-embedding` feature, an
//! OpenAI-compatible `RemoteEmbedder`. Both encode texts into vectors in
//! input order and carry the identity the embedding cache is keyed by.
//!
//! Called by: `topic_modeling::run` and the `.text.embedding` expression.

use std::sync::Arc;

use anyhow::Result;
use tokenizers::Tokenizer;

use super::embedding::{ensure_embedder, Embedder};
#[cfg(feature = "remote-embedding")]
use super::remote_embedding::RemoteEmbedder;
use super::remote_embedding::RemoteEmbeddingOptions;

/// The embedding backend of one run or expression call.
pub enum TextEncoder {
    Local(Arc<Embedder>),
    #[cfg(feature = "remote-embedding")]
    Remote(Arc<RemoteEmbedder>),
}

/// The local model for `model_id`, or a remote encoder for it when `remote`
/// is given. A remote encoder loads the model's metadata, never its ONNX graph.
pub fn ensure_text_encoder(
    model_id: Option<&str>,
    remote: Option<&RemoteEmbeddingOptions>,
) -> Result<TextEncoder> {
    match remote {
        None => Ok(TextEncoder::Local(ensure_embedder(model_id)?)),
        #[cfg(feature = "remote-embedding")]
        Some(options) => Ok(TextEncoder::Remote(Arc::new(RemoteEmbedder::new(
            super::embedding::ensure_model_profile(model_id)?,
            options,
        )?))),
        #[cfg(not(feature = "remote-embedding"))]
        Some(_) => anyhow::bail!(
            "embedding_endpoint needs polars-text built with the remote-embedding feature"
        ),
    }
}

impl TextEncoder {
    /// The local model id; also the remote model's metadata source.
    pub fn model_id(&self) -> &str {
        match self {
            Self::Local(embedder) => embedder.model_id(),
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.model_id(),
        }
    }

    pub fn provider_id(&self) -> &str {
        match self {
            Self::Local(embedder) => embedder.provider_id(),
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.provider_id(),
        }
    }

    pub fn cache_fingerprint(&self) -> &str {
        match self {
            Self::Local(embedder) => embedder.cache_fingerprint(),
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.cache_fingerprint(),
        }
    }

    pub fn max_length(&self) -> usize {
        match self {
            Self::Local(embedder) => embedder.max_length(),
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.max_length(),
        }
    }

    /// The model's tokenizer without truncation or padding, for segmentation.
    #[cfg_attr(not(feature = "topic-modeling"), allow(dead_code))]
    pub fn sizing_tokenizer(&self) -> Tokenizer {
        match self {
            Self::Local(embedder) => embedder.sizing_tokenizer(),
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.sizing_tokenizer(),
        }
    }

    /// Encode `texts` in input order. The local model takes `local_batch`
    /// texts per call; the remote one its own batch size and concurrency.
    /// `on_batch` gets each finished batch's size, on the caller's thread.
    pub fn encode_batched(
        &self,
        texts: &[String],
        local_batch: usize,
        on_batch: &mut dyn FnMut(usize),
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Local(embedder) => {
                let mut vectors = Vec::with_capacity(texts.len());
                for chunk in texts.chunks(local_batch.max(1)) {
                    let encoded = embedder.encode(chunk)?;
                    anyhow::ensure!(
                        encoded.len() == chunk.len(),
                        "embedding batch returned {} vectors for {} texts",
                        encoded.len(),
                        chunk.len()
                    );
                    on_batch(chunk.len());
                    vectors.extend(encoded);
                }
                Ok(vectors)
            }
            #[cfg(feature = "remote-embedding")]
            Self::Remote(remote) => remote.encode_reporting(texts, on_batch),
        }
    }
}
