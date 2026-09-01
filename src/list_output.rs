use std::borrow::Cow;

use polars::prelude::*;
use polars_arrow::array::ListArray;
use polars_arrow::offset::OffsetsBuffer;

/// Construct a List Series over one flat child Series without per-row slicing.
/// Spans must be contiguous and cover the child exactly.
pub fn list_from_spans(
    name: PlSmallStr,
    inner: &Series,
    spans: &[(usize, usize)],
) -> PolarsResult<Series> {
    let mut offsets = Vec::with_capacity(spans.len() + 1);
    offsets.push(0_i64);
    let mut expected_start = 0usize;
    for &(start, end) in spans {
        if start != expected_start || end < start {
            return Err(PolarsError::ComputeError(
                format!("non-contiguous list span ({start}, {end}) after {expected_start}").into(),
            ));
        }
        offsets.push(i64::try_from(end).map_err(|_| {
            PolarsError::ComputeError("list child length exceeds i64 offsets".into())
        })?);
        expected_start = end;
    }
    if expected_start != inner.len() {
        return Err(PolarsError::ComputeError(
            format!(
                "list spans cover {expected_start} child values, but child has {}",
                inner.len()
            )
            .into(),
        ));
    }

    let inner = inner.rechunk();
    if inner.n_chunks() != 1 {
        return Err(PolarsError::ComputeError(
            "failed to consolidate list child into one Arrow array".into(),
        ));
    }
    let compat = CompatLevel::newest();
    let arrow_field = inner.dtype().to_arrow_field(inner.name().clone(), compat);
    let values = inner.to_arrow_with_field(0, Cow::Owned(arrow_field), true)?;
    let list_dtype = DataType::List(Box::new(inner.dtype().clone())).to_arrow(compat);
    let offsets = OffsetsBuffer::<i64>::try_from(offsets)?;
    let array = ListArray::<i64>::try_new(list_dtype, offsets, values, None)?;
    Ok(ListChunked::with_chunk(name, array).into_series())
}
