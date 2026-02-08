use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tokenizer::{ensure_tokenizer, tokenize_text};

pub fn token_frequencies_py(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyAny>> {
    let tokenizer = ensure_tokenizer()
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;

    let mut counts: HashMap<String, u64> = HashMap::new();
    for text in texts {
        if text.trim().is_empty() {
            continue;
        }
        let tokens = tokenize_text(tokenizer, &text, true, true)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;
        for token in tokens {
            *counts.entry(token).or_insert(0) += 1;
        }
    }

    let dict = PyDict::new(py);
    for (token, count) in counts {
        dict.set_item(token, count)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;
    }

    let obj = dict
        .into_pyobject(py)
        .expect("PyDict conversion should not fail");
    Ok(obj.into())
}