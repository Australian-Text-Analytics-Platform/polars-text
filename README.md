# polars-text

Polars 1.44.1 expression plugins for fast, practical text analysis. The
`pl.col("text").text.*` namespace is the sole expression façade; whole-Series
token-frequency and topic-projection utilities remain top-level functions.

## Quick start

```python
import polars as pl
import polars_text

df = pl.DataFrame({
    "text": [
        "Alice said \"Hello world\".",
        "Hello again, world!",
    ]
})

out = df.with_columns([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.char_count().alias("char_count"),
    pl.col("text").text.sentence_count().alias("sentence_count"),
    pl.col("text").text.tokenize(
        model="native:plain_words_en",
        lowercase=True,
        remove_punctuation=True,
    ).alias("tokens"),
])
```

## Expressions and namespace

All expression operations are available through the `text` namespace.

### Tokenization

- `pl.col("text").text.tokenize(model="native:plain_words_en", lowercase=True, remove_punctuation=True, cache=None)`
- `pl.col("text").text.embedding(model=None, cache=None, batch_size=None, embedding_endpoint=None, ...)` (see Remote embeddings)
- `pl.col("text").text.segment_token_counts(unit="automatic", model=None)`
- `pl.col("text").text.clean_text()`
- `pl.col("text").text.word_count()`
- `pl.col("text").text.char_count()`
- `pl.col("text").text.sentence_count()`
- `pl.col("text").text.concordance(query, left_tokens=5, right_tokens=5, regex=False, case_sensitive=False, ignore_punctuation=False)`

### Namespace usage

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

out = df.select([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.tokenize(model="native:plain_words_en").alias("tokens"),
])
```

`tokenize` returns a list of structs with `token`, `start`, and `end`
character offsets. Pass an explicit `native:`, `huggingface:`, or `lindera:`
model ID. Pass `cache=Path("tokens.duckdb")` to persist tokenization results in
a DuckDB cache and reuse them by content hash; leave `cache=None` to compute
directly through the Rust plugin.

A configured cache path is dedicated, disposable `polars-text` storage. Schema
or model-pipeline changes replace the complete DuckDB file; do not put
unrelated user tables in it.

Pass `ignore_punctuation=True` to `concordance` to exclude punctuation and
symbol-only tokens from context counts and L1/R1. The returned contexts retain
the original punctuation and whitespace between lexical tokens and the match;
literal and regular-expression matching are unchanged.

### Embeddings

`embedding` accepts a string expression or a list-of-string expression. String
input returns `List(Float32)` per row; list input returns nested
`List(List(Float32))` per row.

```python
df = pl.DataFrame({"text": ["A short document."], "segments": [["first", "second"]]})

out = df.select([
    pl.col("text").text.embedding(cache="embeddings.duckdb").alias("embedding"),
    pl.col("segments").text.embedding(cache="embeddings.duckdb").alias("segment_embeddings"),
])
```

The Rust plugin downloads and loads Hugging Face ONNX sentence-transformer
repositories automatically through `hf-hub`. Repositories without ONNX files are
not supported. Passing `cache=Path("embeddings.duckdb")` persists vectors in a
separate DuckDB cache keyed by the immutable model snapshot, ONNX artifact,
pooling and normalization graph, canonical maximum length, execution provider,
pipeline version, and text hash. The default model is
`sentence-transformers/all-MiniLM-L6-v2`, whose declared maximum is 256 tokens.

## Concordance

Get left/right context windows around a search term. Output is a list of
structs that you can `explode` and `unnest` for tabular use.

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

concordance = (
    pl.col("text")
    .text.concordance("hello", left_tokens=1, right_tokens=1)
    .list.explode()
    .struct.unnest()
)

out = df.select(concordance)
```

## Topic modelling

`topic_modeling` consumes a complete document column and returns one scalar run
result. The result keeps document outcomes separate from complete topic
metadata:

```text
{
  documents: [{doc_index, dominant_topic, topic_coverage}],
  topics: [{id, representative_words, x, y}],
  n_segments,
  projection_context
}
```

Automatic, Line, and Sentence modes differ only when constructing Topic
Segments. All modes then share embedding, clustering, c-TF-IDF, and document
rollup. Clustering treats every segment as one observation. Rollup weights each
non-overlapping source span by its owned Unicode-character length.

The token budget includes model-added special tokens. Oversized semantic units
are split without overlap or discarded tail text.

- `automatic` packs neighbouring paragraphs into one segment while together
  they fit `max_tokens`; a paragraph is never split to make room, and one over
  the budget on its own splits into sentences, packed the same way within that
  paragraph. Paragraphs are blank-line blocks, or lines when a document has no
  blank line.
- `line` gives every non-empty line its own segment (at most one paragraph per
  segment). A line over the budget is split at the sentence boundary nearest
  its middle, repeatedly, so it becomes a few nearly equal pieces that end at
  full stops.
- `sentence` gives every Unicode sentence its own segment.

HDBSCAN compares every pair of segments, so its time grows with the square of
the segment count. On large corpora, `cluster_sample_size=N` clusters a seeded
sample of N segments (the `seed` argument picks it) and gives every other
segment the topic of its nearest sampled segment by embedding cosine
similarity. `clustered_segments` in the output reports N, or null when every
segment was clustered.

With `max_topic_size=None` (Auto), a topic that is the main topic of more than
half of the documents (a document's main topic owns most of its characters) is
split by re-clustering with a cap below its segment count, as long as most of
its segments stay in topics. `auto_decision` (`not_needed`, `split`, `kept`)
and `auto_document_share` report what happened, and `largest_topic_size` gives
the segments in the largest topic, the scale a fixed `max_topic_size` works on.
Corpora with too little density evidence return no Topics and a null projection
context. Use `project_topics` and `project_topic_basis` with a non-null context
for supported post-fit projections down to one Topic.

### Typical segments of a Topic

From version 0.6.6 a run's projection context also keeps a compact code of
every segment's embedding (its 64 leading principal components, one signed
byte each plus a scale; about 68 bytes per segment).
`polars_text.topic_projection.topic_segment_similarities(context, topic_count,
topic_id)` returns every segment of one (possibly merged) Topic as
`(segment_index, document_index, start_char, end_char, similarity)`, where the
similarity is the cosine to the Topic's centre (the mean of its segments). It
ranks how typical a segment is within one Topic; it is not a probability and
is not comparable across Topics or runs. Older contexts return `None`.

### Progress

`tokenize` and `topic_modeling` take an optional `progress_path`. During the
call they rewrite that JSON file (atomically, at most twice a second):

```json
{"step": 2, "steps": 5, "label": "embedding", "done": 18900,
 "total": 114461, "unit": "segments", "updated_at_ms": 1791428969000}
```

Topic modelling reports five steps: `segmenting` (documents), `embedding`
(segments; cached segments count at once), `arranging`, `grouping` and
`topic_words` (the last three without counts). The tokeniser reports documents
tokenised; every chunk of the column adds to one count, so `total` is null and
the caller supplies it. Polling the file lets an app show progress, since a
Polars plugin cannot call back into Python.

## Token frequencies and stats

Compute corpus token counts and compare corpora with standard statistics.

```python
series_0 = pl.Series("text", ["hello world", "hello again"])
series_1 = pl.Series("text", ["goodbye world"])

freqs_0 = pt.token_frequencies(series_0, model="native:plain_words_en")
freqs_1 = pt.token_frequencies(series_1, model="native:plain_words_en")

stats = pt.token_frequency_stats(freqs_0, freqs_1)
```

## Output schemas

**Tokenization** (list of structs):

- `token`
- `start`
- `end`

**Concordance** (list of structs):

- `left_context`, `matched_text`, `right_context`
- `start_idx`, `end_idx`
- `l1`, `r1` (first token on left/right for quick filtering)

## Models and downloads

Some features download tokenizer assets on first use and run on CPU:

- Hugging Face tokenizers: for example `huggingface:bert-base-uncased`
  (`tokenizer.json` via `hf-hub`)
- Lindera dictionaries: `lindera:cc-cedict`, `lindera:jieba`,
  `lindera:ja-ipadic`, `lindera:ja-ipadic-neologd`, `lindera:ja-unidic`,
  and `lindera:ko-dic` from official Lindera release zips

The initial call may take longer while models download and cache.

`TOKENIZER_MODELS` is the immutable catalogue of
`TokenizerModel(model_id, label, languages)` records.

Embedding features download ONNX artifacts on first use. Some ONNX repositories
store tensor data in sidecar files such as `onnx/model.onnx_data`; those files
are fetched automatically when present.

Embedding sorts the texts of each call by token count and runs them in batches
of 32, so each batch pads only to its own longest text. On macOS the two
curated models (`sentence-transformers/all-MiniLM-L6-v2` and
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) run on Core ML
(GPU and Neural Engine) through fixed-shape sessions, one per length bucket of
32, 64, 128 and 256 tokens up to the model's limit; a batch runs in the
smallest bucket that holds it. Each bucket is compiled on first use and kept in
`coreml/` inside the Hugging Face cache (`HF_HOME`), so only the first run after
an install or upgrade pays the compile. Core ML vectors agree with the CPU's to
about 1e-4. When Core ML cannot be prepared the CPU is used and the reason is
printed once to standard error. Set `POLARS_TEXT_EMBEDDING_PROVIDER=cpu` to keep
a Mac on the CPU. Other models, Linux and Windows embed on ONNX Runtime's CPU
provider with every core. Set `POLARS_TEXT_EMBEDDING_THREADS` to cap the
threads each embedding job (and topic segmentation) uses, for example on a
shared server.

`segment_token_counts` returns, per row, the token counts of the row's units
for a segmentation mode (`automatic`: the whole text; `paragraph`: the
paragraphs Automatic packs; `line`; `sentence`), counted with the embedding
model's tokenizer including its special tokens, without splitting anything. It
loads only the tokenizer. It is meant for choosing a segment token cap from a
sample.

## Remote embeddings

`embedding` and `topic_modeling` can take their sentence embeddings from an
OpenAI-compatible `/v1/embeddings` endpoint instead of the local ONNX model:
for example vLLM on a GPU, usually behind a LiteLLM gateway, from a server pod
or from a desktop without a GPU. Segmentation, clustering, topic words,
coverage and the embedding cache stay local. Without `embedding_endpoint`
nothing changes. Published wheels include it (Cargo feature
`remote-embedding`); minimal builds without it stay network-free.

```python
pl.col("text").text.topic_modeling(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_endpoint="https://gateway.example/v1",
    embedding_api_key_env="GATEWAY_API_KEY",
    embedding_user="person@example.org",
)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `embedding_endpoint` | none (local) | API root including `/v1`; `POST {endpoint}/embeddings` is called. http or https, no credentials, query or fragment. |
| `embedding_api_key_env` | none | Name of an environment variable holding the key, sent as `Authorization: Bearer`. |
| `embedding_api_key_file` | none | Path of a file holding the key or token, read before every request, so a host can keep a short-lived SSO access token fresh during a long run. A 401 or 403 is retried once when the file has changed. Only one of the two key options. |
| `embedding_remote_model` | the local model id | The model name the server knows. |
| `embedding_user` | none | Sent as the OpenAI `user` field, so a gateway such as LiteLLM records each person's use. |
| `embedding_batch_size` | 128 | Texts per request (1 to 2048). |
| `embedding_concurrency` | 4 | Requests in flight (1 to 32); results come back in input order. |
| `embedding_timeout_s` | 120 | Seconds one request may take. |

The key itself is never an option: Polars plans are serialised and logged, so
a key in the plan would leak. Only the variable's name or the file's path is in
the plan.

The local model id (`model` / `embedding_model`) still names the model: its
Sentence Transformers metadata gives the tokenizer used for segmentation, the
length limit (sent as `truncate_prompt_tokens`, which vLLM supports; dropped
for the rest of the run if a server refuses it), the vector width every
response is checked against, and whether vectors are normalised. Normalisation
is applied again on the client, which is harmless and protects against servers
that don't. Only the small metadata files are downloaded; the ONNX graph is not.

Requests are retried with exponential backoff and jitter on connection errors,
`429` (honouring `Retry-After`) and `5xx`, five tries in all; other `4xx` fail
at once. Errors name the host, the HTTP status and the server's message
(shortened), never the key or the input text. A failed remote embedding fails
the run: there is no fallback to the local model, which would mix two embedding
spaces in one run. HTTPS trusts the computer's own certificates, so it works on
networks that inspect TLS.

Remote vectors have their own cache scope (`remote:<host>/<served model>` plus
the normalisation and truncation settings): they are never mixed with local
vectors, even for the same model, since tiny numeric differences can change
clusters. A server for testing:

```bash
vllm serve sentence-transformers/all-MiniLM-L6-v2 --runner pooling --port 8010
vllm serve sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --runner pooling --port 8011
```

With `POLARS_TEXT_TEST_EMBEDDING_ENDPOINT` (and
`POLARS_TEXT_TEST_EMBEDDING_ENDPOINT_MULTILINGUAL` when the multilingual model
is served elsewhere; `POLARS_TEXT_TEST_EMBEDDING_API_KEY_ENV` names a key
variable if needed), `cargo test parity_with_a_real_server -- --nocapture`
compares local and remote vectors for both models and reports topic
differences.

## Development

Build the extension locally with maturin and then import as `polars_text`.
See the repository-level
[development](../docs/runbooks/polars-text-development.md) and
[release](../docs/runbooks/polars-text-release.md) runbooks for complete
procedures.

```bash
make build
make test
```

For faster Rust iteration, use feature-scoped targets such as
`make check-tokenization`, `make build-tokenization`, or `make build-topic`.
Leave `JOBS` unset for Cargo's default parallelism, or pass `JOBS=<n>` to cap it.
