use polars::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ConcordanceKwargs {
    pub search_word: String,
    pub num_left_tokens: i64,
    pub num_right_tokens: i64,
    pub regex: bool,
    pub case_sensitive: bool,
    #[serde(default)]
    pub ignore_punctuation: bool,
}

pub fn list_struct_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(concordance_struct_type())),
    ))
}

pub fn concordance_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("left_context".into(), DataType::String),
        Field::new("matched_text".into(), DataType::String),
        Field::new("right_context".into(), DataType::String),
        Field::new("start_idx".into(), DataType::Int64),
        Field::new("end_idx".into(), DataType::Int64),
        Field::new("l1".into(), DataType::String),
        Field::new("r1".into(), DataType::String),
    ])
}

fn empty_struct_series() -> PolarsResult<Series> {
    let fields = [
        Series::new("left_context".into(), Vec::<String>::new()),
        Series::new("matched_text".into(), Vec::<String>::new()),
        Series::new("right_context".into(), Vec::<String>::new()),
        Series::new("start_idx".into(), Vec::<i64>::new()),
        Series::new("end_idx".into(), Vec::<i64>::new()),
        Series::new("l1".into(), Vec::<String>::new()),
        Series::new("r1".into(), Vec::<String>::new()),
    ];
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, 0, fields.iter())?.into_series())
}

pub fn struct_series_from_matches(
    matches: Vec<ldaca_rs::text::ConcordanceMatch>,
) -> PolarsResult<Series> {
    if matches.is_empty() {
        return empty_struct_series();
    }
    let columns = [
        Series::new(
            "left_context".into(),
            matches
                .iter()
                .map(|m| m.left_context.as_str())
                .collect::<Vec<_>>(),
        ),
        Series::new(
            "matched_text".into(),
            matches
                .iter()
                .map(|m| m.matched_text.as_str())
                .collect::<Vec<_>>(),
        ),
        Series::new(
            "right_context".into(),
            matches
                .iter()
                .map(|m| m.right_context.as_str())
                .collect::<Vec<_>>(),
        ),
        Series::new(
            "start_idx".into(),
            matches.iter().map(|m| m.start_idx).collect::<Vec<_>>(),
        ),
        Series::new(
            "end_idx".into(),
            matches.iter().map(|m| m.end_idx).collect::<Vec<_>>(),
        ),
        Series::new(
            "l1".into(),
            matches.iter().map(|m| m.l1.as_str()).collect::<Vec<_>>(),
        ),
        Series::new(
            "r1".into(),
            matches.iter().map(|m| m.r1.as_str()).collect::<Vec<_>>(),
        ),
    ];
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, matches.len(), columns.iter())?.into_series())
}
