#[cfg(feature = "tokenization")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
#[cfg(feature = "tokenization")]
use pyo3_polars::PySeries;
use std::collections::HashMap;

#[cfg(feature = "tokenization")]
pub fn token_frequencies_py(
    py: Python<'_>,
    series: PySeries,
    model_id: &str,
) -> PyResult<Py<PyAny>> {
    let model_id = model_id.to_owned();
    let counts = py
        .detach(move || {
            let texts = series.0.str().map_err(|e| e.to_string())?;
            let tokenizer =
                ldaca_rs::text::Tokenizer::load(&model_id).map_err(|e| e.to_string())?;
            ldaca_rs::text::token_frequencies(texts.iter().flatten(), &tokenizer)
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
    let dict = PyDict::new(py);
    for (token, count) in counts {
        dict.set_item(token, count)?;
    }
    Ok(dict.unbind().into_any())
}

#[pyfunction]
pub fn frequency_stats(
    py: Python<'_>,
    corpus_0: HashMap<String, u64>,
    corpus_1: HashMap<String, u64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = py
        .detach(move || {
            ldaca_rs::text::frequency_stats(&corpus_0, &corpus_1).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;
    rows.into_iter()
        .map(|row| {
            let dict = PyDict::new(py);
            macro_rules! field {
                ($name:ident) => {
                    dict.set_item(stringify!($name), row.$name)?;
                };
            }
            field!(token);
            macro_rules! integer {
                ($name:ident) => {
                    dict.set_item(
                        stringify!($name),
                        i64::try_from(row.$name).map_err(|_| {
                            PyValueError::new_err("frequency exceeds Int64 output range")
                        })?,
                    )?;
                };
            }
            integer!(freq_corpus_0);
            integer!(freq_corpus_1);
            integer!(corpus_0_total);
            integer!(corpus_1_total);
            field!(expected_0);
            field!(expected_1);
            field!(log_likelihood_llv);
            field!(bayes_factor_bic);
            field!(effect_size_ell);
            field!(significance);
            field!(percent_corpus_0);
            field!(percent_corpus_1);
            field!(percent_diff);
            field!(relative_risk);
            field!(log_ratio);
            field!(odds_ratio);
            Ok(dict.unbind())
        })
        .collect()
}
