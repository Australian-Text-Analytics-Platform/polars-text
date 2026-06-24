use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;
use pyo3_polars::PolarsAllocator;

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "tokenization")]
mod concordance;
pub mod expressions;
#[cfg(feature = "tokenization")]
mod lindera_dict;
#[cfg(feature = "tokenization")]
mod offsets;
#[cfg(feature = "tokenization")]
mod token_frequencies;
#[cfg(feature = "tokenization")]
mod tokenizer;
#[cfg(any(feature = "embedding", feature = "topic-modeling"))]
pub mod topic_modeling;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(compiled_features, _m)?)?;
    _m.add_function(wrap_pyfunction!(token_frequencies_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(prefetch_tokenizer_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(loaded_tokenizers_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(debug_token_cache_snapshot_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(prefetch_embedder_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(loaded_embedders_py, _m)?)?;
    Ok(())
}

fn compiled_feature_names() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "full") {
        features.push("full");
    }
    if cfg!(feature = "cache") {
        features.push("cache");
    }
    if cfg!(feature = "tokenization") {
        features.push("tokenization");
    }
    if cfg!(feature = "embedding") {
        features.push("embedding");
    }
    if cfg!(feature = "topic-modeling") {
        features.push("topic-modeling");
    }
    features
}

#[pyfunction]
fn compiled_features(py: Python<'_>) -> PyResult<Bound<'_, PyFrozenSet>> {
    PyFrozenSet::new(py, compiled_feature_names())
}

#[cfg(any(not(feature = "tokenization"), not(feature = "embedding")))]
fn feature_disabled(operation: &str, feature: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "{operation} requires the '{feature}' feature; rebuild polars-text with that feature or use the default full build"
    ))
}

#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (texts, model))]
fn token_frequencies_py(py: Python<'_>, texts: Vec<String>, model: String) -> PyResult<Py<PyAny>> {
    token_frequencies_py_impl(py, texts, model)
}

#[cfg(feature = "tokenization")]
fn token_frequencies_py_impl(
    py: Python<'_>,
    texts: Vec<String>,
    model: String,
) -> PyResult<Py<PyAny>> {
    token_frequencies::token_frequencies_py(py, texts, Some(model.as_str()))
}

#[cfg(not(feature = "tokenization"))]
fn token_frequencies_py_impl(
    _py: Python<'_>,
    _texts: Vec<String>,
    _model: String,
) -> PyResult<Py<PyAny>> {
    Err(feature_disabled("token_frequencies", "tokenization"))
}

#[pyfunction(name = "prefetch_tokenizer")]
#[pyo3(signature = (model_id))]
fn prefetch_tokenizer_py(model_id: &str) -> PyResult<()> {
    prefetch_tokenizer_py_impl(model_id)
}

#[cfg(feature = "tokenization")]
fn prefetch_tokenizer_py_impl(model_id: &str) -> PyResult<()> {
    tokenizer::ensure_tokenizer_for_model(Some(model_id))
        .map(|_| ())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

#[cfg(not(feature = "tokenization"))]
fn prefetch_tokenizer_py_impl(_model_id: &str) -> PyResult<()> {
    Err(feature_disabled("prefetch_tokenizer", "tokenization"))
}

#[pyfunction(name = "loaded_tokenizers")]
fn loaded_tokenizers_py() -> Vec<String> {
    loaded_tokenizers_py_impl()
}

#[cfg(feature = "tokenization")]
fn loaded_tokenizers_py_impl() -> Vec<String> {
    tokenizer::loaded_model_ids()
}

#[cfg(not(feature = "tokenization"))]
fn loaded_tokenizers_py_impl() -> Vec<String> {
    Vec::new()
}

#[pyfunction(name = "debug_token_cache_snapshot")]
#[pyo3(signature = (path))]
fn debug_token_cache_snapshot_py(
    path: PathBuf,
) -> PyResult<(Vec<String>, Vec<expressions::TokenCacheDebugRow>)> {
    debug_token_cache_snapshot_py_impl(path)
}

#[cfg(feature = "tokenization")]
fn debug_token_cache_snapshot_py_impl(
    path: PathBuf,
) -> PyResult<(Vec<String>, Vec<expressions::TokenCacheDebugRow>)> {
    expressions::debug_token_cache_snapshot(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

#[cfg(not(feature = "tokenization"))]
fn debug_token_cache_snapshot_py_impl(
    _path: PathBuf,
) -> PyResult<(Vec<String>, Vec<expressions::TokenCacheDebugRow>)> {
    Err(feature_disabled(
        "debug_token_cache_snapshot",
        "tokenization",
    ))
}

/// Download/load the ONNX Runtime embedder for `repo_id` (default model if `None`) so
/// a later `run_topic_modeling` call doesn't pay the load cost. Mirrors
/// `prefetch_tokenizer`.
#[pyfunction(name = "prefetch_embedder")]
#[pyo3(signature = (repo_id=None))]
fn prefetch_embedder_py(repo_id: Option<String>) -> PyResult<()> {
    prefetch_embedder_py_impl(repo_id)
}

#[cfg(feature = "embedding")]
fn prefetch_embedder_py_impl(repo_id: Option<String>) -> PyResult<()> {
    topic_modeling::embedding::ensure_embedder(repo_id.as_deref())
        .map(|_| ())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

#[cfg(not(feature = "embedding"))]
fn prefetch_embedder_py_impl(_repo_id: Option<String>) -> PyResult<()> {
    Err(feature_disabled("prefetch_embedder", "embedding"))
}

/// Repo ids of embedders currently resident in the in-process registry.
#[pyfunction(name = "loaded_embedders")]
fn loaded_embedders_py() -> Vec<String> {
    loaded_embedders_py_impl()
}

#[cfg(feature = "embedding")]
fn loaded_embedders_py_impl() -> Vec<String> {
    topic_modeling::embedding::loaded_embedder_ids()
}

#[cfg(not(feature = "embedding"))]
fn loaded_embedders_py_impl() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_feature_names_include_every_enabled_feature() {
        let features = compiled_feature_names();

        assert_eq!(features.contains(&"full"), cfg!(feature = "full"));
        assert_eq!(features.contains(&"cache"), cfg!(feature = "cache"));
        assert_eq!(
            features.contains(&"tokenization"),
            cfg!(feature = "tokenization")
        );
        assert_eq!(features.contains(&"embedding"), cfg!(feature = "embedding"));
        assert_eq!(
            features.contains(&"topic-modeling"),
            cfg!(feature = "topic-modeling")
        );
    }
}
