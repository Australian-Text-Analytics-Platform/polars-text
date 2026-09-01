use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;
use pyo3_polars::PolarsAllocator;
#[cfg(feature = "tokenization")]
use pyo3_polars::PySeries;

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "tokenization")]
mod concordance;
pub mod expressions;
#[cfg(feature = "tokenization")]
mod lindera_dict;
#[cfg(any(feature = "embedding", feature = "tokenization"))]
mod list_output;
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
fn _internal(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compiled_features, module)?)?;
    #[cfg(feature = "tokenization")]
    module.add_function(wrap_pyfunction!(token_frequencies_py, module)?)?;
    #[cfg(feature = "topic-modeling")]
    {
        module.add_function(wrap_pyfunction!(project_topic_modeling_context_py, module)?)?;
        module.add_function(wrap_pyfunction!(project_topic_modeling_basis_py, module)?)?;
    }
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

#[cfg(feature = "tokenization")]
#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (series, model))]
fn token_frequencies_py(py: Python<'_>, series: PySeries, model: String) -> PyResult<Py<PyAny>> {
    token_frequencies::token_frequencies_py(py, series, Some(model.as_str()))
}

#[cfg(feature = "topic-modeling")]
#[pyfunction(name = "project_topic_modeling_context")]
#[pyo3(signature = (context, cluster_count))]
fn project_topic_modeling_context_py(context: Vec<u8>, cluster_count: usize) -> PyResult<String> {
    let result = topic_modeling::projection::project_serialized_context(&context, cluster_count)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:#}")))?;
    serde_json::to_string(&result)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(format!("{error:#}")))
}

#[cfg(feature = "topic-modeling")]
#[pyfunction(name = "project_topic_modeling_basis")]
#[pyo3(signature = (context, cluster_count, corpus_sizes))]
fn project_topic_modeling_basis_py(
    context: Vec<u8>,
    cluster_count: usize,
    corpus_sizes: Vec<usize>,
) -> PyResult<String> {
    let basis = topic_modeling::projection::project_serialized_context_basis(
        &context,
        cluster_count,
        &corpus_sizes,
    )
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:#}")))?;
    serde_json::to_string(&basis)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(format!("{error:#}")))
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
