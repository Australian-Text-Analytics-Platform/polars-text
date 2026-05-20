# Tokenizers And Plan Paths

## Tokenizer Registry

`src/tokenizer.rs` owns a process-local registry:

```rust
OnceCell<RwLock<HashMap<String, Arc<TokenizerBackend>>>>
```

`ensure_tokenizer_for_model()` first checks the registry, then loads the
backend on a cache miss. Loaded backends are shared by `Arc`.

## Backends

`TokenizerBackend` supports:

- Hugging Face `tokenizers::Tokenizer`,
- `jieba-rs` for Chinese word segmentation,
- Lindera for Japanese and Korean morphological segmentation.

Hugging Face models are loaded from `tokenizer.json` through `hf-hub`. Jieba is
local. Lindera dictionaries are downloaded on first use.

## Lindera Dictionaries

`src/lindera_dict.rs` downloads prebuilt dictionary tarballs from the Hugging
Face dataset `SIH/lindera-dicts`, unless `LDACA_LINDERA_DICT_REPO` overrides
the repo for tests.

The cache is stored under the OS cache directory in `ldaca/lindera/`. A lock
file prevents simultaneous extract races. A dictionary is considered complete
when `matrix.mtx` exists in the extracted directory.

## Case And Offset Handling

Only Hugging Face tokenizers are case-aware. Jieba and Lindera operate on
scripts where case-folding is not meaningful, so the lowercase branch returns a
borrowed string instead of allocating a lowercase copy.

Hugging Face and Lindera emit byte offsets. The backend and UI use character
offsets. `offsets.rs` converts monotonic byte spans to character spans in one
forward pass, reducing per-document work from repeated scans to `O(chars +
tokens)`.

## Plain Tokenizer

`tokenize_plain_text()` uses the BERT pre-tokenizer and removes punctuation or
special tokens when requested. It supports token-frequency counting and
concordance context tokenization without loading a full model backend.

## Serialized Plan Paths

Polars `.plbin` files can store absolute file paths for scan nodes. Moving a
workspace would normally break those plans.

`src/plan_paths.rs` deserializes a `DslPlan`, walks the tree, and inspects or
rewrites `DslPlan::Scan` sources:

- `list_source_paths(path)` returns scan source paths in depth-first order.
- `replace_source_paths(path, mapper)` applies exact path substitutions and
  writes the plan back only if something changed.

The Cargo feature set for Polars plan deserialization must include all plan
variants the app can serialize. Tests cover random sample and joined scans
because missing Polars features surface as plan-deserialization errors.
