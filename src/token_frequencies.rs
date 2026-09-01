use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PySeries;

use crate::tokenizer::ensure_tokenizer_for_model;

pub fn token_frequencies_py(
    py: Python<'_>,
    series: PySeries,
    model_id: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let model_id = model_id.map(str::to_owned);
    let counts = py
        .detach(move || -> Result<HashMap<String, u64>, String> {
            let texts = series
                .0
                .str()
                .map_err(|error| format!("token_frequencies expects a String Series: {error}"))?;
            let backend = ensure_tokenizer_for_model(model_id.as_deref())
                .map_err(|error| error.to_string())?;
            let mut counts = HashMap::new();
            for text in texts.iter().flatten() {
                if text.trim().is_empty() {
                    continue;
                }
                for token in backend
                    .tokenize_text(text, false, true, true)
                    .map_err(|error| error.to_string())?
                {
                    *counts.entry(token).or_insert(0) += 1;
                }
            }
            Ok(counts)
        })
        .map_err(PyRuntimeError::new_err)?;

    let dict = PyDict::new(py);
    for (token, count) in counts {
        dict.set_item(token, count)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;
    }

    Ok(dict.unbind().into_any())
}
