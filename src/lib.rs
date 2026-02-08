use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

pub mod expressions;
mod tokenizer;
mod concordance;
mod topic_modeling;
mod quotation;
mod pos_tagging;
mod token_frequencies;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
	_m.add_function(wrap_pyfunction!(topic_modeling_py, _m)?)?;
	_m.add_function(wrap_pyfunction!(token_frequencies_py, _m)?)?;
	Ok(())
}

#[pyfunction(name = "topic_modeling")]
#[pyo3(signature = (texts, min_points=5, eps=None, max_terms=5, seed=42))]
fn topic_modeling_py(
	py: Python<'_>,
	texts: Vec<String>,
	min_points: usize,
	eps: Option<f32>,
	max_terms: usize,
	seed: u64,
) -> PyResult<(Py<PyAny>, Vec<Vec<(i64, f32)>>)> {
	topic_modeling::topic_modeling_py(py, texts, min_points, eps, max_terms, seed)
}

#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (texts))]
fn token_frequencies_py(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyAny>> {
	token_frequencies::token_frequencies_py(py, texts)
}
