//! Remote embedding against an in-process mock server: no network needed.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::Tokenizer;

use super::client::{scrubbed_message, RemoteEmbedder, RetryPolicy};
use super::RemoteEmbeddingOptions;
use crate::topic_modeling::embedding::ModelProfile;
use crate::topic_modeling::embedding_cache::{get_or_insert_embeddings, CacheScope};

/// One request as the server saw it.
#[derive(Clone, Debug)]
struct Seen {
    authorization: Option<String>,
    body: Value,
}

struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
}

fn ok(body: Value) -> Reply {
    Reply {
        status: 200,
        headers: Vec::new(),
        body: body.to_string(),
    }
}

fn status(status: u16, body: Value) -> Reply {
    Reply {
        status,
        headers: Vec::new(),
        body: body.to_string(),
    }
}

type Handler = dyn Fn(usize, &Seen) -> Reply + Send + Sync;

/// An HTTP/1.1 server answering every request with `handler(n, request)`,
/// where `n` counts requests from 0. Returns its `/v1` root and the log.
fn mock_server(handler: Arc<Handler>) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = format!("http://{}/v1", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let handler = Arc::clone(&handler);
            let log = Arc::clone(&log);
            std::thread::spawn(move || serve(stream, &*handler, &log));
        }
    });
    (root, seen)
}

fn serve(stream: TcpStream, handler: &Handler, log: &Mutex<Vec<Seen>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut length = 0usize;
    let mut authorization = None;
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
            break;
        }
        let (name, value) = line.split_once(':').unwrap();
        let value = value.trim().to_string();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.parse().unwrap(),
            "authorization" => authorization = Some(value),
            _ => {}
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let request = Seen {
        authorization,
        body: serde_json::from_slice(&body).unwrap(),
    };
    let index = {
        let mut log = log.lock().unwrap();
        log.push(request.clone());
        log.len() - 1
    };
    let reply = handler(index, &request);
    let mut stream = stream;
    let extra = reply
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect::<String>();
    let _ = write!(
        stream,
        "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n{}",
        reply.status,
        reply.body.len(),
        reply.body
    );
}

/// Embeddings for a request: `vector(text)` per input, listed in reverse
/// order with their indices, as a server may.
fn reversed_embeddings(request: &Seen, vector: impl Fn(&str) -> Vec<f32>) -> Value {
    let inputs = request.body["input"].as_array().unwrap();
    let data = inputs
        .iter()
        .enumerate()
        .rev()
        .map(|(index, text)| json!({"index": index, "embedding": vector(text.as_str().unwrap())}))
        .collect::<Vec<_>>();
    json!({"object": "list", "data": data, "model": request.body["model"]})
}

fn profile(normalize: bool) -> Arc<ModelProfile> {
    let model = WordLevel::builder()
        .vocab([("[UNK]".to_string(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    Arc::new(ModelProfile {
        model_id: "sentence-transformers/test-model".into(),
        revision: "rev-a".into(),
        tokenizer: Tokenizer::new(model),
        max_length: 8,
        normalize,
        embedding_dim: 3,
    })
}

fn options(endpoint: &str) -> RemoteEmbeddingOptions {
    RemoteEmbeddingOptions {
        endpoint: endpoint.into(),
        api_key_env: None,
        api_key_file: None,
        model: None,
        user: None,
        batch_size: 3,
        concurrency: 3,
        timeout_s: 10.0,
    }
}

const FAST: RetryPolicy = RetryPolicy {
    attempts: 3,
    base: Duration::from_millis(1),
    cap: Duration::from_millis(5),
    retry_after_cap: Duration::from_secs(5),
};

fn texts(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("text {index}")).collect()
}

/// `[1, n, 0]` for "text n": the direction keeps n after normalising.
fn numbered(text: &str) -> Vec<f32> {
    vec![1.0, text[5..].parse::<f32>().unwrap(), 0.0]
}

#[test]
fn batches_run_concurrently_and_come_back_in_input_order() {
    let (root, seen) = mock_server(Arc::new(|_, request| {
        ok(reversed_embeddings(request, numbered))
    }));
    let remote = RemoteEmbedder::with_retry(
        profile(false),
        &RemoteEmbeddingOptions {
            model: Some("served/minilm".into()),
            user: Some("person-7".into()),
            ..options(&root)
        },
        FAST,
    )
    .unwrap();
    let caller = std::thread::current().id();
    let mut reported = Vec::new();
    let vectors = remote
        .encode_reporting(&texts(10), &mut |count| {
            assert_eq!(std::thread::current().id(), caller);
            reported.push(count);
        })
        .unwrap();

    for (index, vector) in vectors.iter().enumerate() {
        assert_eq!(vector, &numbered(&format!("text {index}")));
    }
    reported.sort_unstable();
    assert_eq!(reported, vec![1, 3, 3, 3]);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    for request in seen.iter() {
        assert_eq!(request.body["model"], "served/minilm");
        assert_eq!(request.body["user"], "person-7");
        assert_eq!(request.body["encoding_format"], "float");
        assert_eq!(request.body["truncate_prompt_tokens"], 8);
        assert!(request.body["input"].as_array().unwrap().len() <= 3);
        assert_eq!(request.authorization, None);
    }
    assert!(remote.provider_id().starts_with("remote:127.0.0.1:"));
    assert!(remote.provider_id().ends_with("/served/minilm"));
}

#[test]
fn the_served_model_defaults_to_the_local_model_id() {
    let (root, seen) = mock_server(Arc::new(|_, request| {
        ok(reversed_embeddings(request, |_| vec![1.0, 0.0, 0.0]))
    }));
    let remote = RemoteEmbedder::with_retry(profile(true), &options(&root), FAST).unwrap();
    remote.encode_reporting(&texts(1), &mut |_| {}).unwrap();
    assert_eq!(
        seen.lock().unwrap()[0].body["model"],
        "sentence-transformers/test-model"
    );
}

#[test]
fn vectors_are_normalised_when_the_model_normalises() {
    let (root, _) = mock_server(Arc::new(|_, request| {
        ok(reversed_embeddings(request, |_| vec![3.0, 4.0, 0.0]))
    }));
    let normalised = RemoteEmbedder::with_retry(profile(true), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(2), &mut |_| {})
        .unwrap();
    assert_eq!(normalised, vec![vec![0.6, 0.8, 0.0]; 2]);

    let raw = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(2), &mut |_| {})
        .unwrap();
    assert_eq!(raw, vec![vec![3.0, 4.0, 0.0]; 2]);
}

#[test]
fn rate_limits_wait_for_retry_after_then_succeed() {
    let (root, seen) = mock_server(Arc::new(|index, request| {
        if index == 0 {
            Reply {
                status: 429,
                headers: vec![("Retry-After", "1".into())],
                body: json!({"error": {"message": "slow down"}}).to_string(),
            }
        } else {
            ok(reversed_embeddings(request, numbered))
        }
    }));
    let remote = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST).unwrap();
    let started = Instant::now();
    let vectors = remote.encode_reporting(&texts(2), &mut |_| {}).unwrap();
    assert!(started.elapsed() >= Duration::from_secs(1));
    assert_eq!(vectors.len(), 2);
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[test]
fn server_errors_are_retried_until_the_attempts_run_out() {
    let (root, seen) = mock_server(Arc::new(|_, _| {
        status(500, json!({"error": {"message": "GPU out of memory"}}))
    }));
    let remote = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST).unwrap();
    let error = remote
        .encode_reporting(&texts(2), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("HTTP 500"), "{error}");
    assert!(error.contains("GPU out of memory"), "{error}");
    assert!(error.contains("gave up after 3 attempts"), "{error}");
    assert_eq!(seen.lock().unwrap().len(), 3);
}

#[test]
fn other_client_errors_are_not_retried() {
    let (root, seen) = mock_server(Arc::new(|_, _| {
        status(404, json!({"detail": "model not found"}))
    }));
    let remote = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST).unwrap();
    let error = remote
        .encode_reporting(&texts(2), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("HTTP 404: model not found"), "{error}");
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn unreachable_servers_are_retried_then_reported() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let remote = RemoteEmbedder::with_retry(
        profile(false),
        &options(&format!("http://127.0.0.1:{port}/v1")),
        FAST,
    )
    .unwrap();
    let error = remote
        .encode_reporting(&texts(1), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("could not be reached"), "{error}");
    assert!(error.contains("gave up after 3 attempts"), "{error}");
}

#[test]
fn a_wrong_vector_count_or_width_fails_the_run() {
    let (root, _) = mock_server(Arc::new(|_, request| {
        let mut body = reversed_embeddings(request, |_| vec![1.0, 0.0, 0.0]);
        body["data"].as_array_mut().unwrap().pop();
        ok(body)
    }));
    let error = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(3), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("returned 2 vectors for 3 texts"), "{error}");

    let (root, _) = mock_server(Arc::new(|_, request| {
        ok(reversed_embeddings(request, |_| vec![1.0, 0.0, 0.0, 0.0]))
    }));
    let error = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(3), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("4-dimensional vectors"), "{error}");
    assert!(error.contains("has 3"), "{error}");
}

#[test]
fn a_server_that_refuses_truncation_gets_requests_without_it() {
    let (root, seen) = mock_server(Arc::new(|_, request| {
        if request.body.get("truncate_prompt_tokens").is_some() {
            status(
                400,
                json!({"error": {"message": "Unrecognized request argument supplied: truncate_prompt_tokens"}}),
            )
        } else {
            ok(reversed_embeddings(request, numbered))
        }
    }));
    let remote = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST).unwrap();
    remote.encode_reporting(&texts(2), &mut |_| {}).unwrap();
    remote.encode_reporting(&texts(2), &mut |_| {}).unwrap();
    let seen = seen.lock().unwrap();
    let with_field = seen
        .iter()
        .filter(|request| request.body.get("truncate_prompt_tokens").is_some())
        .count();
    assert_eq!((seen.len(), with_field), (3, 1));
}

#[test]
fn refused_keys_and_sign_in_redirects_say_what_they_mean() {
    let (root, seen) = mock_server(Arc::new(|_, _| status(401, json!({}))));
    let error = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(1), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("refused the key (HTTP 401)"), "{error}");
    assert!(error.contains("single sign-on"), "{error}");
    assert_eq!(seen.lock().unwrap().len(), 1);

    let (root, seen) = mock_server(Arc::new(|_, _| Reply {
        status: 302,
        headers: vec![("Location", "https://sso.example/auth".into())],
        body: String::new(),
    }));
    let error = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST)
        .unwrap()
        .encode_reporting(&texts(1), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("redirected the request (HTTP 302)"),
        "{error}"
    );
    assert!(!error.contains("sso.example"), "{error}");
    // Never followed to the sign-in page.
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn errors_never_contain_the_key_or_the_input() {
    let key = "sk-test-very-secret-key-123";
    std::env::set_var("POLARS_TEXT_TEST_REMOTE_KEY_SCRUB", key);
    let private = "A very private sentence about someone's health records";
    let (root, seen) = mock_server(Arc::new(move |_, _| {
        status(
            400,
            json!({"error": {"message": format!("bad key {key}; cannot embed '{private}'")}}),
        )
    }));
    let remote = RemoteEmbedder::with_retry(
        profile(false),
        &RemoteEmbeddingOptions {
            api_key_env: Some("POLARS_TEXT_TEST_REMOTE_KEY_SCRUB".into()),
            ..options(&root)
        },
        FAST,
    )
    .unwrap();
    let error = remote
        .encode_reporting(&[private.to_string()], &mut |_| {})
        .unwrap_err();
    let rendered = format!("{error:#} {error:?}");
    assert!(!rendered.contains(key), "{rendered}");
    assert!(!rendered.contains("private sentence"), "{rendered}");
    assert!(rendered.contains("HTTP 400"), "{rendered}");
    assert_eq!(
        seen.lock().unwrap()[0].authorization.as_deref(),
        Some(format!("Bearer {key}").as_str())
    );

    let short = scrubbed_message(
        &json!({"error": "unknown word 'zebra pancakes'"}).to_string(),
        &["zebra pancakes".to_string()],
        None,
    );
    assert_eq!(short, "unknown word '[input text]'");
}

#[test]
fn a_token_refreshed_in_its_file_is_used_after_a_401() {
    let directory = tempfile::tempdir().unwrap();
    let token = directory.path().join("token");
    std::fs::write(&token, "old-token\n").unwrap();
    let refreshed = token.clone();
    let (root, seen) = mock_server(Arc::new(move |_, request| {
        if request.authorization.as_deref() == Some("Bearer old-token") {
            // The host refreshes the SSO token while the request is out.
            std::fs::write(&refreshed, "new-token").unwrap();
            status(401, json!({"error": {"message": "token expired"}}))
        } else {
            ok(reversed_embeddings(request, numbered))
        }
    }));
    let remote = RemoteEmbedder::with_retry(
        profile(false),
        &RemoteEmbeddingOptions {
            api_key_file: Some(token.to_string_lossy().into_owned()),
            concurrency: 1,
            ..options(&root)
        },
        FAST,
    )
    .unwrap();
    remote.encode_reporting(&texts(2), &mut |_| {}).unwrap();
    let authorizations = seen
        .lock()
        .unwrap()
        .iter()
        .map(|request| request.authorization.clone().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(authorizations, ["Bearer old-token", "Bearer new-token"]);
}

#[test]
fn remote_and_local_vectors_never_share_cache_entries() {
    let (root, _) = mock_server(Arc::new(|_, request| {
        ok(reversed_embeddings(request, numbered))
    }));
    let remote = RemoteEmbedder::with_retry(profile(false), &options(&root), FAST).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let cache = directory.path().join("embeddings.duckdb");
    let inputs = texts(3);
    let local = CacheScope {
        model_id: "sentence-transformers/test-model",
        fingerprint: "local-fingerprint",
        provider_id: "CPUExecutionProvider",
    };
    let remote_scope = CacheScope {
        model_id: remote.model_id(),
        fingerprint: remote.cache_fingerprint(),
        provider_id: remote.provider_id(),
    };
    let mut computed = 0;
    for scope in [local, remote_scope, local, remote_scope] {
        get_or_insert_embeddings(&cache, scope, &inputs, |misses| {
            computed += misses.len();
            Ok(misses.iter().map(|text| numbered(text)).collect())
        })
        .unwrap();
    }
    // Each scope computes the texts once; neither reuses the other's.
    assert_eq!(computed, 6);
    assert_ne!(remote.cache_fingerprint(), "local-fingerprint");
}

#[cfg(feature = "topic-modeling")]
/// Sentences across several subjects and, for the multilingual model,
/// languages: enough variety for a meaningful parity check.
fn parity_sentences(multilingual: bool) -> Vec<String> {
    let subjects = [
        (
            "Doctors",
            "treated patients in the hospital",
            "医生在医院治疗病人",
        ),
        (
            "Trains",
            "ran late on the northern line",
            "火车在北线晚点了",
        ),
        (
            "The team",
            "won the final in extra time",
            "球队在加时赛中赢得了决赛",
        ),
        (
            "Farmers",
            "planted wheat after the rain",
            "农民在雨后种植小麦",
        ),
        (
            "The council",
            "approved a budget for new housing",
            "市议会批准了新住房预算",
        ),
        (
            "Storms",
            "closed schools along the coast",
            "暴风雨导致沿海学校停课",
        ),
    ];
    let frames = [
        "Yesterday, {s} {p}.",
        "Reports say {s} {p} again this week.",
        "{s} {p}, according to officials.",
        "Few expected that {s} {p}.",
        "In {y}, {s} {p}.",
    ];
    let mut sentences = Vec::new();
    for year in 2015..2025 {
        for (subject, predicate, chinese) in subjects {
            for frame in frames {
                sentences.push(
                    frame
                        .replace("{s}", subject)
                        .replace("{p}", predicate)
                        .replace("{y}", &year.to_string()),
                );
            }
            if multilingual {
                sentences.push(format!("{year}年，{chinese}。"));
            }
        }
    }
    sentences.truncate(if multilingual { 360 } else { 300 });
    sentences
}

/// Parity with a real server (vLLM, or a gateway in front of one), for both
/// MiniLM models. Runs only with `POLARS_TEXT_TEST_EMBEDDING_ENDPOINT` set
/// (and `POLARS_TEXT_TEST_EMBEDDING_ENDPOINT_MULTILINGUAL` when the
/// multilingual model is served elsewhere); `POLARS_TEXT_TEST_EMBEDDING_API_KEY_ENV`
/// names the variable holding a key, if the server needs one. Asserts a
/// cosine above 0.999 per sentence, and reports topic differences rather
/// than requiring identity.
#[cfg(feature = "topic-modeling")]
#[test]
fn parity_with_a_real_server() -> anyhow::Result<()> {
    use crate::topic_modeling::embedding::{
        ensure_embedder, ensure_model_profile, DEFAULT_EMBEDDER_REPO_ID,
        MULTILINGUAL_EMBEDDER_REPO_ID,
    };
    use crate::topic_modeling::segmentation::SegmentationConfig;
    use crate::topic_modeling::{run, RunConfig};

    let Ok(endpoint) = std::env::var("POLARS_TEXT_TEST_EMBEDDING_ENDPOINT") else {
        eprintln!("skipped: set POLARS_TEXT_TEST_EMBEDDING_ENDPOINT to run remote parity");
        return Ok(());
    };
    let multilingual_endpoint = std::env::var("POLARS_TEXT_TEST_EMBEDDING_ENDPOINT_MULTILINGUAL")
        .unwrap_or_else(|_| endpoint.clone());
    let api_key_env = std::env::var("POLARS_TEXT_TEST_EMBEDDING_API_KEY_ENV").ok();
    for (model, endpoint, multilingual) in [
        (DEFAULT_EMBEDDER_REPO_ID, endpoint.clone(), false),
        (MULTILINGUAL_EMBEDDER_REPO_ID, multilingual_endpoint, true),
    ] {
        let options = RemoteEmbeddingOptions {
            endpoint,
            api_key_env: api_key_env.clone(),
            batch_size: 64,
            concurrency: 4,
            timeout_s: 120.0,
            ..options("http://unused/v1")
        };
        let sentences = parity_sentences(multilingual);
        let local = ensure_embedder(Some(model))?.encode(&sentences)?;
        let remote = RemoteEmbedder::new(ensure_model_profile(Some(model))?, &options)?
            .encode_reporting(&sentences, &mut |_| {})?;
        let cosines = local
            .iter()
            .zip(&remote)
            .map(|(left, right)| {
                let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
                let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
                dot / (norm(left) * norm(right))
            })
            .collect::<Vec<_>>();
        let worst = cosines.iter().copied().fold(1.0_f32, f32::min);
        eprintln!(
            "PARITY {model}: {} sentences, lowest cosine {worst:.6}",
            sentences.len()
        );
        assert!(worst > 0.999, "{model}: lowest cosine {worst}");

        // Topic modelling both ways on a small corpus: report, don't require identity.
        let documents = sentences
            .chunks(3)
            .map(|chunk| chunk.join(" "))
            .collect::<Vec<_>>();
        let documents = documents.iter().map(String::as_str).collect::<Vec<_>>();
        let config = |remote_embedding| RunConfig {
            embedder_repo_id: Some(model.to_string()),
            embedding_cache_path: None,
            segmentation: SegmentationConfig {
                max_tokens: if multilingual { 128 } else { 256 },
                ..SegmentationConfig::default()
            },
            seed: 0,
            min_cluster_size: 5,
            max_cluster_size: None,
            vectorizer_model_id: None,
            lowercase: true,
            cluster_sample_size: None,
            progress_path: None,
            remote_embedding,
        };
        let summary = |result: &crate::topic_modeling::TopicModelingResult| {
            let mut sizes = std::collections::HashMap::new();
            for document in &result.documents {
                *sizes.entry(document.dominant_topic).or_insert(0usize) += 1;
            }
            let mut topics = result
                .topics
                .iter()
                .map(|topic| {
                    (
                        sizes.get(&topic.id).copied().unwrap_or(0),
                        topic
                            .representative_words
                            .iter()
                            .take(5)
                            .map(|word| word.word.clone())
                            .collect::<Vec<_>>()
                            .join(" "),
                    )
                })
                .collect::<Vec<_>>();
            topics.sort_by_key(|topic| std::cmp::Reverse(topic.0));
            topics.truncate(5);
            (result.topics.len(), topics)
        };
        let local_run = summary(&run(&documents, &config(None))?);
        let remote_run = summary(&run(&documents, &config(Some(options.clone())))?);
        eprintln!(
            "TOPICS {model}: local {} topics, remote {} topics",
            local_run.0, remote_run.0
        );
        for (index, (left, right)) in local_run.1.iter().zip(&remote_run.1).enumerate() {
            let same = if left.1 == right.1 {
                "same"
            } else {
                "DIFFERENT"
            };
            eprintln!(
                "  #{index} local ({} docs) [{}] | remote ({} docs) [{}] {same}",
                left.0, left.1, right.0, right.1
            );
        }
    }
    Ok(())
}
