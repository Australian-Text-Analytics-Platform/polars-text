//! Sentence embeddings from an OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Why: embedding is the slow part of topic modelling on a CPU. A remote GPU
//! (vLLM, usually behind a LiteLLM gateway that also counts each person's
//! use) embeds a large corpus in seconds, for a server pod or a desktop
//! without a GPU. Everything else (segmentation, clustering, topic words,
//! the embedding cache) stays local.
//!
//! The local model id still names the model: its Sentence Transformers
//! metadata gives the tokenizer (segmentation), the length limit
//! (`truncate_prompt_tokens`), whether vectors are normalised, and their
//! width, which every response is checked against. Only the ONNX graph is
//! not loaded.
//!
//! Keys never travel in the plan: Polars plans are serialised and logged, so
//! the options name an environment variable (`embedding_api_key_env`) or a
//! file (`embedding_api_key_file`) that holds the key. Both are read before
//! every request, so a host can keep a short-lived SSO access token fresh in
//! the file during a long run. Errors name the host, the HTTP status and the
//! server's message, never the key or the input text.
//!
//! There is no fallback to the local model: one run must use one embedding
//! space, so a failed remote run fails.

use anyhow::{Context, Result};
use serde::Deserialize;

/// Texts per request when the host doesn't say.
pub const DEFAULT_REMOTE_BATCH_SIZE: usize = 128;
/// Requests in flight when the host doesn't say.
pub const DEFAULT_REMOTE_CONCURRENCY: usize = 4;
/// Seconds one request may take when the host doesn't say.
pub const DEFAULT_REMOTE_TIMEOUT_S: f64 = 120.0;
const MAX_REMOTE_BATCH_SIZE: usize = 2048;
const MAX_REMOTE_CONCURRENCY: usize = 32;

/// Remote embedding settings, as both plugins receive them. Every field is
/// optional, so plans built without them deserialise as before; they take
/// effect only with `embedding_endpoint`.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct RemoteEmbeddingKwargs {
    #[serde(default)]
    pub embedding_endpoint: Option<String>,
    #[serde(default)]
    pub embedding_api_key_env: Option<String>,
    #[serde(default)]
    pub embedding_api_key_file: Option<String>,
    #[serde(default)]
    pub embedding_remote_model: Option<String>,
    #[serde(default)]
    pub embedding_user: Option<String>,
    #[serde(default)]
    pub embedding_batch_size: Option<usize>,
    #[serde(default)]
    pub embedding_concurrency: Option<usize>,
    #[serde(default)]
    pub embedding_timeout_s: Option<f64>,
}

/// Validated remote embedding settings.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteEmbeddingOptions {
    /// API root including `/v1`, for example `http://gateway:4000/v1`.
    pub endpoint: String,
    pub api_key_env: Option<String>,
    pub api_key_file: Option<String>,
    /// The model name the server knows; the local model id when unset.
    pub model: Option<String>,
    /// Sent as the OpenAI `user` field, for per-person accounting.
    pub user: Option<String>,
    pub batch_size: usize,
    pub concurrency: usize,
    pub timeout_s: f64,
}

impl RemoteEmbeddingKwargs {
    /// The validated options, or `None` without an endpoint (local embedding).
    pub fn into_options(self) -> Result<Option<RemoteEmbeddingOptions>> {
        let Some(endpoint) = non_empty(self.embedding_endpoint) else {
            let others = [
                (
                    "embedding_api_key_env",
                    self.embedding_api_key_env.is_some(),
                ),
                (
                    "embedding_api_key_file",
                    self.embedding_api_key_file.is_some(),
                ),
                (
                    "embedding_remote_model",
                    self.embedding_remote_model.is_some(),
                ),
                ("embedding_user", self.embedding_user.is_some()),
                ("embedding_batch_size", self.embedding_batch_size.is_some()),
                (
                    "embedding_concurrency",
                    self.embedding_concurrency.is_some(),
                ),
                ("embedding_timeout_s", self.embedding_timeout_s.is_some()),
            ];
            if let Some((name, _)) = others.iter().find(|(_, set)| *set) {
                anyhow::bail!("{name} needs embedding_endpoint");
            }
            return Ok(None);
        };
        let options = RemoteEmbeddingOptions {
            endpoint,
            api_key_env: non_empty(self.embedding_api_key_env),
            api_key_file: non_empty(self.embedding_api_key_file),
            model: non_empty(self.embedding_remote_model),
            user: non_empty(self.embedding_user),
            batch_size: self
                .embedding_batch_size
                .unwrap_or(DEFAULT_REMOTE_BATCH_SIZE),
            concurrency: self
                .embedding_concurrency
                .unwrap_or(DEFAULT_REMOTE_CONCURRENCY),
            timeout_s: self.embedding_timeout_s.unwrap_or(DEFAULT_REMOTE_TIMEOUT_S),
        };
        options.validate()?;
        Ok(Some(options))
    }
}

impl RemoteEmbeddingOptions {
    /// Checks every setting; errors name the setting, never its value.
    pub fn validate(&self) -> Result<()> {
        parse_endpoint(&self.endpoint)?;
        anyhow::ensure!(
            (1..=MAX_REMOTE_BATCH_SIZE).contains(&self.batch_size),
            "embedding_batch_size must be between 1 and {MAX_REMOTE_BATCH_SIZE}"
        );
        anyhow::ensure!(
            (1..=MAX_REMOTE_CONCURRENCY).contains(&self.concurrency),
            "embedding_concurrency must be between 1 and {MAX_REMOTE_CONCURRENCY}"
        );
        anyhow::ensure!(
            self.timeout_s.is_finite() && self.timeout_s > 0.0,
            "embedding_timeout_s must be a positive number of seconds"
        );
        Credential::from_options(self)?.read()?;
        Ok(())
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// A parsed endpoint: the `/embeddings` URL and the host for ids and errors.
#[derive(Debug, Clone, PartialEq)]
struct Endpoint {
    /// The API root without a trailing slash.
    base: String,
    /// `host[:port]`, lower-cased.
    host: String,
}

fn parse_endpoint(raw: &str) -> Result<Endpoint> {
    const SHAPE: &str =
        "embedding_endpoint must be an http or https URL such as https://gateway.example/v1";
    let raw = raw.trim();
    let (scheme, rest) = raw.split_once("://").context(SHAPE)?;
    anyhow::ensure!(
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"),
        SHAPE
    );
    anyhow::ensure!(
        !rest.contains(['?', '#']),
        "embedding_endpoint must not have a query or fragment; give the key with embedding_api_key_env or embedding_api_key_file"
    );
    let authority = rest.split('/').next().unwrap_or_default();
    anyhow::ensure!(
        !authority.contains('@'),
        "embedding_endpoint must not contain credentials; give the key with embedding_api_key_env or embedding_api_key_file"
    );
    anyhow::ensure!(
        !authority.is_empty() && !authority.contains(char::is_whitespace),
        SHAPE
    );
    Ok(Endpoint {
        base: format!(
            "{}://{}",
            scheme.to_ascii_lowercase(),
            rest.trim_end_matches('/')
        ),
        host: authority.to_ascii_lowercase(),
    })
}

/// Where the key comes from; read again before every request.
#[derive(Debug, Clone, PartialEq)]
enum Credential {
    None,
    Env(String),
    File(std::path::PathBuf),
}

impl Credential {
    fn from_options(options: &RemoteEmbeddingOptions) -> Result<Self> {
        match (&options.api_key_env, &options.api_key_file) {
            (Some(_), Some(_)) => {
                anyhow::bail!("set only one of embedding_api_key_env and embedding_api_key_file")
            }
            (Some(name), None) => Ok(Self::Env(name.clone())),
            (None, Some(path)) => Ok(Self::File(path.into())),
            (None, None) => Ok(Self::None),
        }
    }

    /// The current key; never part of an error message.
    fn read(&self) -> Result<Option<String>> {
        match self {
            Self::None => Ok(None),
            Self::Env(name) => std::env::var(name)
                .ok()
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
                .map(Some)
                .context("the environment variable named by embedding_api_key_env is not set"),
            Self::File(path) => std::fs::read_to_string(path)
                .ok()
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
                .map(Some)
                .context("the file named by embedding_api_key_file could not be read or is empty"),
        }
    }

    #[cfg_attr(not(feature = "remote-embedding"), allow(dead_code))]
    fn refreshable(&self) -> bool {
        matches!(self, Self::File(_))
    }
}

#[cfg(feature = "remote-embedding")]
pub use client::RemoteEmbedder;

#[cfg(feature = "remote-embedding")]
mod client {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    use anyhow::Result;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use tokenizers::Tokenizer;

    use super::{parse_endpoint, Credential, RemoteEmbeddingOptions};
    use crate::topic_modeling::embedding::{
        normalize_nested_rows, tokenizer_for_sizing, ModelProfile,
    };

    /// Largest response body read (a 2,048 x 1,024 float batch is about 25 MB).
    const MAX_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;
    /// Characters of a server's message kept in an error.
    const MAX_MESSAGE_CHARS: usize = 300;
    /// A server message containing this many leading characters of an input
    /// text is withheld: it quotes the input.
    const QUOTED_INPUT_CHARS: usize = 24;
    const TRUNCATION_FIELD: &str = "truncate_prompt_tokens";

    #[derive(Debug, Clone, Copy)]
    pub(super) struct RetryPolicy {
        /// Tries per batch, the first included.
        pub attempts: u32,
        pub base: Duration,
        pub cap: Duration,
        pub retry_after_cap: Duration,
    }

    impl Default for RetryPolicy {
        fn default() -> Self {
            Self {
                attempts: 5,
                base: Duration::from_millis(500),
                cap: Duration::from_secs(20),
                retry_after_cap: Duration::from_secs(120),
            }
        }
    }

    impl RetryPolicy {
        /// `Retry-After` when the server gave one, else exponential backoff
        /// with jitter (between half and all of the step).
        fn delay(&self, failures: u32, retry_after: Option<Duration>) -> Duration {
            if let Some(wait) = retry_after {
                return wait.min(self.retry_after_cap);
            }
            let step = self
                .base
                .saturating_mul(1u32 << failures.saturating_sub(1).min(16))
                .min(self.cap);
            step.mul_f64(0.5 + 0.5 * jitter())
        }
    }

    /// A fraction in [0, 1) that differs between threads and moments.
    fn jitter() -> f64 {
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos()),
        );
        (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// An OpenAI-compatible embeddings client for one model.
    pub struct RemoteEmbedder {
        profile: Arc<ModelProfile>,
        embeddings_url: String,
        host: String,
        served_model: String,
        credential: Credential,
        user: Option<String>,
        batch_size: usize,
        concurrency: usize,
        agent: ureq::Agent,
        retry: RetryPolicy,
        /// Cleared for the rest of the run when the server refuses the field.
        send_truncation: AtomicBool,
        provider_id: String,
        cache_fingerprint: String,
    }

    struct Reply {
        status: u16,
        retry_after: Option<Duration>,
        body: String,
    }

    #[derive(Deserialize)]
    struct EmbeddingsResponse {
        data: Vec<EmbeddingDatum>,
    }

    #[derive(Deserialize)]
    struct EmbeddingDatum {
        index: usize,
        embedding: Vec<f32>,
    }

    impl RemoteEmbedder {
        pub(crate) fn new(
            profile: Arc<ModelProfile>,
            options: &RemoteEmbeddingOptions,
        ) -> Result<Self> {
            Self::with_retry(profile, options, RetryPolicy::default())
        }

        pub(super) fn with_retry(
            profile: Arc<ModelProfile>,
            options: &RemoteEmbeddingOptions,
            retry: RetryPolicy,
        ) -> Result<Self> {
            options.validate()?;
            let endpoint = parse_endpoint(&options.endpoint)?;
            let credential = Credential::from_options(options)?;
            let served_model = options
                .model
                .clone()
                .unwrap_or_else(|| profile.model_id.clone());
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs_f64(options.timeout_s)))
                .http_status_as_error(false)
                // A redirect is reported, never followed: behind single
                // sign-on it leads to a login page, not to embeddings.
                .max_redirects(0)
                .tls_config(
                    ureq::tls::TlsConfig::builder()
                        .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                        .build(),
                )
                .build()
                .into();
            let provider_id = format!("remote:{}/{served_model}", endpoint.host);
            let cache_fingerprint =
                remote_fingerprint(&profile, &endpoint.base, &served_model, &provider_id);
            Ok(Self {
                embeddings_url: format!("{}/embeddings", endpoint.base),
                host: endpoint.host,
                served_model,
                credential,
                user: options.user.clone(),
                batch_size: options.batch_size,
                concurrency: options.concurrency,
                agent,
                retry,
                send_truncation: AtomicBool::new(true),
                provider_id,
                cache_fingerprint,
                profile,
            })
        }

        /// The local model id whose metadata this encoder follows.
        pub fn model_id(&self) -> &str {
            &self.profile.model_id
        }

        /// `remote:<host>/<served model>`: the cache scope of remote vectors.
        pub fn provider_id(&self) -> &str {
            &self.provider_id
        }

        pub fn cache_fingerprint(&self) -> &str {
            &self.cache_fingerprint
        }

        pub fn max_length(&self) -> usize {
            self.profile.max_length
        }

        pub fn sizing_tokenizer(&self) -> Tokenizer {
            tokenizer_for_sizing(&self.profile.tokenizer)
        }

        /// Encode `texts` in input order, `embedding_batch_size` texts per
        /// request and up to `embedding_concurrency` requests in flight.
        /// `on_batch` runs on the caller's thread as each request finishes.
        /// The first failure stops the run: no batch falls back to the CPU.
        pub fn encode_reporting(
            &self,
            texts: &[String],
            on_batch: &mut dyn FnMut(usize),
        ) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let batches = texts.chunks(self.batch_size).collect::<Vec<_>>();
            let workers = self.concurrency.min(batches.len());
            let next = AtomicUsize::new(0);
            let stop = AtomicBool::new(false);
            let mut results: Vec<Option<Vec<Vec<f32>>>> = vec![None; batches.len()];
            let mut first_error = None;
            std::thread::scope(|scope| {
                let (sender, receiver) = mpsc::channel();
                for _ in 0..workers {
                    let sender = sender.clone();
                    let (next, stop, batches) = (&next, &stop, &batches);
                    scope.spawn(move || loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(batch) = batches.get(index) else {
                            break;
                        };
                        let result = self.request_batch(batch);
                        if result.is_err() {
                            stop.store(true, Ordering::Relaxed);
                        }
                        if sender.send((index, result)).is_err() {
                            break;
                        }
                    });
                }
                drop(sender);
                for (index, result) in receiver {
                    match result {
                        Ok(vectors) => {
                            on_batch(vectors.len());
                            results[index] = Some(vectors);
                        }
                        Err(error) => {
                            stop.store(true, Ordering::Relaxed);
                            first_error.get_or_insert(error);
                        }
                    }
                }
            });
            if let Some(error) = first_error {
                return Err(error);
            }
            let mut vectors = Vec::with_capacity(texts.len());
            for batch in results {
                vectors.extend(batch.ok_or_else(|| {
                    anyhow::anyhow!("remote embedding at {} left a batch unanswered", self.host)
                })?);
            }
            Ok(vectors)
        }

        /// One request, with retries on connection errors, 429 and 5xx.
        fn request_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut failures = 0u32;
            let mut refreshed_credential = false;
            loop {
                let truncate = self.send_truncation.load(Ordering::Relaxed);
                let body = self.request_body(texts, truncate)?;
                let key = self.credential.read()?;
                let (message, retry_after) = match self.post(&body, key.as_deref()) {
                    Err(error) => {
                        let message = format!(
                            "remote embedding at {} could not be reached: {error}",
                            self.host
                        );
                        if !transport_error_is_retryable(&error) {
                            anyhow::bail!(message);
                        }
                        (message, None)
                    }
                    Ok(reply) if (200..300).contains(&reply.status) => {
                        return self.parse(&reply.body, texts);
                    }
                    Ok(reply) => {
                        if truncate
                            && (400..500).contains(&reply.status)
                            && reply.status != 429
                            && reply.body.contains(TRUNCATION_FIELD)
                        {
                            // The server doesn't know the field: send without it.
                            self.send_truncation.store(false, Ordering::Relaxed);
                            continue;
                        }
                        if matches!(reply.status, 401 | 403)
                            && self.credential.refreshable()
                            && !refreshed_credential
                        {
                            // A token refreshed by the host since the request
                            // was sent gets one more try.
                            refreshed_credential = true;
                            if self.credential.read()? != key {
                                continue;
                            }
                        }
                        let message = self.status_error(&reply, texts, key.as_deref());
                        if reply.status != 429 && reply.status < 500 {
                            anyhow::bail!(message);
                        }
                        (message, reply.retry_after)
                    }
                };
                failures += 1;
                if failures >= self.retry.attempts {
                    anyhow::bail!("{message} (gave up after {failures} attempts)");
                }
                std::thread::sleep(self.retry.delay(failures, retry_after));
            }
        }

        fn request_body(&self, texts: &[String], truncate: bool) -> Result<String> {
            let mut body = serde_json::json!({
                "model": self.served_model,
                "input": texts,
                "encoding_format": "float",
            });
            if let Some(user) = &self.user {
                body["user"] = serde_json::Value::from(user.as_str());
            }
            if truncate {
                body[TRUNCATION_FIELD] = serde_json::Value::from(self.profile.max_length);
            }
            Ok(serde_json::to_string(&body)?)
        }

        fn post(&self, body: &str, key: Option<&str>) -> Result<Reply, ureq::Error> {
            let mut request = self
                .agent
                .post(&self.embeddings_url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json");
            if let Some(key) = key {
                request = request.header("Authorization", format!("Bearer {key}"));
            }
            let mut response = request.send(body)?;
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<u64>().ok())
                .map(Duration::from_secs);
            let body = response
                .body_mut()
                .with_config()
                .limit(MAX_RESPONSE_BYTES)
                .read_to_string()?;
            Ok(Reply {
                status,
                retry_after,
                body,
            })
        }

        /// Vectors in input order, checked and normalised as the local model would.
        fn parse(&self, body: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let host = &self.host;
            let response: EmbeddingsResponse = serde_json::from_str(body).map_err(|error| {
                anyhow::anyhow!(
                    "remote embedding at {host} answered with something other than embeddings ({})",
                    error.classify_message()
                )
            })?;
            anyhow::ensure!(
                response.data.len() == texts.len(),
                "remote embedding at {host} returned {} vectors for {} texts",
                response.data.len(),
                texts.len()
            );
            let dimension = self.profile.embedding_dim;
            let mut ordered: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
            for datum in response.data {
                let slot = ordered.get_mut(datum.index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "remote embedding at {host} returned an index outside the batch"
                    )
                })?;
                anyhow::ensure!(
                    slot.is_none(),
                    "remote embedding at {host} returned the same index twice"
                );
                anyhow::ensure!(
                    datum.embedding.len() == dimension,
                    "remote embedding at {host} returned {}-dimensional vectors, but {} has {dimension}; check embedding_remote_model",
                    datum.embedding.len(),
                    self.profile.model_id
                );
                anyhow::ensure!(
                    datum.embedding.iter().all(|value| value.is_finite()),
                    "remote embedding at {host} returned a vector with non-finite values"
                );
                *slot = Some(datum.embedding);
            }
            let mut rows = ordered.into_iter().flatten().collect::<Vec<_>>();
            if self.profile.normalize {
                // Idempotent, and protects against servers that don't normalise.
                normalize_nested_rows(&mut rows);
            }
            Ok(rows)
        }

        /// `remote embedding at <host> answered HTTP <status>: <message>`,
        /// without the key or any input text. Refused keys and sign-in
        /// redirects say what they mean.
        fn status_error(&self, reply: &Reply, texts: &[String], key: Option<&str>) -> String {
            let host = &self.host;
            let status = reply.status;
            let message = scrubbed_message(&reply.body, texts, key);
            let said = if message == NO_MESSAGE {
                String::new()
            } else {
                format!(": {message}")
            };
            match status {
                300..=399 => format!(
                    "remote embedding at {host} redirected the request (HTTP {status}), \
                     probably to a sign-in page: this endpoint needs single sign-on, \
                     not an API key"
                ),
                401 | 403 => format!(
                    "remote embedding at {host} refused the key (HTTP {status}){}",
                    if said.is_empty() {
                        ". Check the key; an endpoint behind single sign-on doesn't accept API keys"
                            .to_string()
                    } else {
                        said
                    }
                ),
                _ => format!("remote embedding at {host} answered HTTP {status}{said}"),
            }
        }
    }

    trait ClassifyMessage {
        fn classify_message(&self) -> &'static str;
    }

    impl ClassifyMessage for serde_json::Error {
        /// The kind of parse failure, never the response content.
        fn classify_message(&self) -> &'static str {
            match self.classify() {
                serde_json::error::Category::Io => "unreadable body",
                serde_json::error::Category::Syntax => "not JSON",
                serde_json::error::Category::Data => "unexpected JSON shape",
                serde_json::error::Category::Eof => "truncated JSON",
            }
        }
    }

    fn transport_error_is_retryable(error: &ureq::Error) -> bool {
        matches!(
            error,
            ureq::Error::Io(_)
                | ureq::Error::Timeout(_)
                | ureq::Error::ConnectionFailed
                | ureq::Error::HostNotFound
                | ureq::Error::Protocol(_)
                | ureq::Error::BodyStalled
        )
    }

    const NO_MESSAGE: &str = "(no message)";

    /// The server's own message (OpenAI `error.message`, FastAPI `detail`,
    /// or the body), shortened, with the key and quoted input removed.
    pub(super) fn scrubbed_message(body: &str, texts: &[String], key: Option<&str>) -> String {
        let parsed = serde_json::from_str::<serde_json::Value>(body).ok();
        if parsed
            .as_ref()
            .is_some_and(|value| value.as_object().is_some_and(serde_json::Map::is_empty))
        {
            return NO_MESSAGE.into();
        }
        let message = parsed
            .as_ref()
            .and_then(|value| {
                value
                    .pointer("/error/message")
                    .or_else(|| value.get("error"))
                    .or_else(|| value.get("detail"))
                    .or_else(|| value.get("message"))
            })
            .map(|value| match value {
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| body.to_string());
        let mut message = message.split_whitespace().collect::<Vec<_>>().join(" ");
        if let Some(key) = key.filter(|key| key.len() >= 4) {
            message = message.replace(key, "[key]");
        }
        for text in texts {
            let text = text.trim();
            let prefix = text.chars().take(QUOTED_INPUT_CHARS).collect::<String>();
            if prefix.chars().count() == QUOTED_INPUT_CHARS && message.contains(&prefix) {
                return "(the server's message quotes the input, so it is not shown)".into();
            }
            if text.chars().count() >= 6 {
                message = message.replace(text, "[input text]");
            }
        }
        if message.is_empty() {
            return NO_MESSAGE.into();
        }
        if message.chars().count() > MAX_MESSAGE_CHARS {
            message = message.chars().take(MAX_MESSAGE_CHARS).collect::<String>() + "…";
        }
        message
    }

    /// Cache identity of remote vectors: never shared with local ones, even
    /// for the same model, since small numeric differences change clusters.
    fn remote_fingerprint(
        profile: &ModelProfile,
        endpoint: &str,
        served_model: &str,
        provider_id: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        for value in [
            "remote-embedding-v1".to_string(),
            profile.model_id.clone(),
            profile.revision.clone(),
            endpoint.to_string(),
            served_model.to_string(),
            provider_id.to_string(),
            format!("normalize={}", profile.normalize),
            format!("truncate={}", profile.max_length),
            profile.embedding_dim.to_string(),
        ] {
            hasher.update(value.as_bytes());
            hasher.update([0]);
        }
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kwargs(endpoint: &str) -> RemoteEmbeddingKwargs {
        RemoteEmbeddingKwargs {
            embedding_endpoint: Some(endpoint.into()),
            ..Default::default()
        }
    }

    #[test]
    fn no_endpoint_means_local_and_other_options_need_one() {
        assert_eq!(
            RemoteEmbeddingKwargs::default().into_options().unwrap(),
            None
        );
        let error = RemoteEmbeddingKwargs {
            embedding_user: Some("ab12".into()),
            ..Default::default()
        }
        .into_options()
        .unwrap_err();
        assert_eq!(error.to_string(), "embedding_user needs embedding_endpoint");
    }

    #[test]
    fn options_take_defaults() {
        let options = kwargs(" http://gateway.example:4000/v1/ ")
            .into_options()
            .unwrap()
            .unwrap();
        assert_eq!(options.batch_size, DEFAULT_REMOTE_BATCH_SIZE);
        assert_eq!(options.concurrency, DEFAULT_REMOTE_CONCURRENCY);
        assert_eq!(options.timeout_s, DEFAULT_REMOTE_TIMEOUT_S);
        assert_eq!(
            parse_endpoint(&options.endpoint).unwrap(),
            Endpoint {
                base: "http://gateway.example:4000/v1".into(),
                host: "gateway.example:4000".into(),
            }
        );
    }

    #[test]
    fn endpoint_errors_name_the_setting_never_its_value() {
        for (endpoint, expected) in [
            ("ftp://secret-host/v1", "http or https"),
            ("gateway/v1", "http or https"),
            ("https://user:hunter2@secret-host/v1", "credentials"),
            ("https://secret-host/v1?key=hunter2", "query or fragment"),
            ("https:///v1", "http or https"),
        ] {
            let message = kwargs(endpoint).into_options().unwrap_err().to_string();
            assert!(message.contains(expected), "{endpoint}: {message}");
            assert!(!message.contains("secret-host") && !message.contains("hunter2"));
        }
    }

    #[test]
    fn key_sources_are_checked_and_exclusive() {
        let missing = RemoteEmbeddingKwargs {
            embedding_api_key_env: Some("POLARS_TEXT_TEST_SURELY_UNSET_KEY".into()),
            ..kwargs("https://gateway.example/v1")
        };
        let message = missing.into_options().unwrap_err().to_string();
        assert!(message.contains("embedding_api_key_env"));
        assert!(!message.contains("POLARS_TEXT_TEST_SURELY_UNSET_KEY"));

        let both = RemoteEmbeddingKwargs {
            embedding_api_key_env: Some("A".into()),
            embedding_api_key_file: Some("/b".into()),
            ..kwargs("https://gateway.example/v1")
        };
        assert!(both
            .into_options()
            .unwrap_err()
            .to_string()
            .contains("only one"));

        let file = RemoteEmbeddingKwargs {
            embedding_api_key_file: Some("/surely/missing/token".into()),
            ..kwargs("https://gateway.example/v1")
        };
        let message = file.into_options().unwrap_err().to_string();
        assert!(message.contains("embedding_api_key_file") && !message.contains("/surely"));
    }

    #[test]
    fn tuning_limits_are_checked() {
        for (kwargs, setting) in [
            (
                RemoteEmbeddingKwargs {
                    embedding_batch_size: Some(0),
                    ..kwargs("http://h/v1")
                },
                "embedding_batch_size",
            ),
            (
                RemoteEmbeddingKwargs {
                    embedding_concurrency: Some(33),
                    ..kwargs("http://h/v1")
                },
                "embedding_concurrency",
            ),
            (
                RemoteEmbeddingKwargs {
                    embedding_timeout_s: Some(0.0),
                    ..kwargs("http://h/v1")
                },
                "embedding_timeout_s",
            ),
        ] {
            assert!(kwargs
                .into_options()
                .unwrap_err()
                .to_string()
                .contains(setting));
        }
    }
}

#[cfg(all(test, feature = "remote-embedding"))]
#[path = "remote_embedding/client_tests.rs"]
mod client_tests;
