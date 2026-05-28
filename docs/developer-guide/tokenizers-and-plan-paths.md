# Tokenizers

## Tokenizer Registry

`src/tokenizer.rs` owns a process-local registry:

```rust
OnceCell<RwLock<HashMap<String, Arc<TokenizerBackend>>>>
```

`ensure_tokenizer_for_model()` first checks the registry, then loads the
backend on a cache miss. Loaded backends are shared by `Arc`.

## Backends

`TokenizerBackend` supports:

- `native:*` IDs for in-process tokenizers. Today this is only
  `native:plain_words_en` for local rule-based English word tokenization.
- `huggingface:*` IDs for Hugging Face `tokenizers::Tokenizer` repos. The
  prefix is stripped before the repo ID is passed to `hf-hub`; for example
  `huggingface:bert-base-uncased` downloads `tokenizer.json` from the
  `bert-base-uncased` model repo.
- `lindera:*` IDs for Chinese CC-CEDICT / Jieba word segmentation, Japanese
  IPADIC / IPADIC-neologd / UniDic morphological segmentation, and Korean
  ko-dic morphological segmentation.

`native:plain_words_en` is stateless and uses the BERT pre-tokenizer without loading a
model. Hugging Face models are loaded from `tokenizer.json` through `hf-hub`.
No Lindera dictionaries are embedded in the extension. Every `lindera:*`
dictionary is downloaded on first use.

## Lindera Dictionaries

`src/lindera_dict.rs` downloads official prebuilt dictionary zips from the
Lindera GitHub Releases page on first use.

The cache is stored under `$HOME/.cache/ldaca/`, or under `LINDERA_DICT_PATH`
when that environment variable is set. A lock file prevents simultaneous
extract races. A dictionary is considered complete when `matrix.mtx` exists in
the extracted directory.

Lindera's own `LINDERA_CACHE` and `LINDERA_DICTIONARIES_PATH` variables are
used by Lindera dictionary crates at build time. At runtime, when embedded
dictionaries are disabled, Lindera's `load_dictionary` API accepts an
`embedded://` URI, a `file://` URI, or a filesystem path. Because this build
does not embed dictionaries, `polars-text` must resolve an on-disk dictionary
directory and pass that concrete path to Lindera.

## Case And Offset Handling

Only `native:plain_words_en` and Hugging Face tokenizers are case-aware. Lindera
backends operate on scripts where case-folding is not meaningful, so the
lowercase branch returns a borrowed string instead of allocating a lowercase
copy.

Hugging Face and Lindera emit byte offsets. The backend and UI use character
offsets. `offsets.rs` converts monotonic byte spans to character spans in one
forward pass, reducing per-document work from repeated scans to `O(chars +
tokens)`.

## Plain Tokenizer

`native:plain_words_en` and the legacy `tokenize_plain_text()` wrapper use the BERT
pre-tokenizer and remove punctuation or special tokens when requested. Token
frequency counting now routes through the same backend dispatch as
`.text.tokenize(model=...)`; the legacy helper remains for concordance context
tokenization.

## Serialized Plan Paths

Polars `.plbin` source-path listing and rewriting moved to the sibling
`polars-source-utils` package. That package carries the broad `polars-plan`
feature surface needed for plan deserialization; `polars-text` only keeps the
text-processing plugin and tokenizer code.
