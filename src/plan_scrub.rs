//! Surgically remove targeted aliased FFI-plugin expressions from a
//! serialized Polars `LazyFrame` plan.
//!
//! Polars' query optimiser does NOT dead-code eliminate FFI plugin
//! calls — plugins may have side effects, and indeed our
//! `tokenize_with_cache_lookup` does (it writes a fresh delta-cache
//! parquet whenever it sees any row miss the in-memory cache map).
//!
//! That means the natural-looking sequence
//!
//! ```text
//!     lf.drop("__derived__.tokens.text.jieba", strict=False)
//!       .with_columns(<new tokenize_with_cache_lookup expression>
//!                     .alias("__derived__.tokens.text.jieba"))
//! ```
//!
//! does *not* replace the old plugin call. It leaves the old `HStack`
//! node alive — its output column is hidden by a subsequent `Select`,
//! but the expression itself still fires on every `collect()` and still
//! writes a parquet to whatever cache directory its old `user_id`
//! kwarg points at. In a multi-user setup that means a workspace
//! imported by user B will keep writing cache parquets into user A's
//! tree forever.
//!
//! `scrub_plugin_expressions` walks the `DslPlan` and removes any
//! `Expr::Alias(<inner>, <name>)` from `HStack` / `Select` where:
//!
//! * `<name>` matches one of the requested alias names, AND
//! * `<inner>` is a `FunctionExpr::FfiPlugin` whose `symbol` matches
//!   the requested plugin symbol.
//!
//! The double check (alias name AND plugin symbol) is intentional. The
//! alias-name match alone is sufficient in our well-behaved internal
//! flow, but the plugin-symbol guard keeps the function safe to use
//! against arbitrary user plans without risk of accidentally tearing
//! out a non-plugin expression that happens to share a name. The pair
//! of checks is also what makes "scrub all tokens-cache lookups in a
//! plan" cheap to reason about.
//!
//! Returns the number of expressions removed (zero if there was
//! nothing to scrub, which keeps the call cheap on plans that have
//! never been tokenised).

use std::io::Cursor;
use std::sync::Arc;

use polars_plan::dsl::{DslPlan, Expr, FunctionExpr, PlanSerializationContext};

fn deserialize_plan(bytes: &[u8]) -> Result<DslPlan, String> {
    DslPlan::deserialize_versioned(Cursor::new(bytes))
        .map_err(|e| format!("scrub: failed to deserialize plan: {e}"))
}

fn serialize_plan(plan: &DslPlan) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    plan.serialize_versioned(&mut buf, PlanSerializationContext::default())
        .map_err(|e| format!("scrub: failed to serialize plan: {e}"))?;
    Ok(buf)
}

/// True if `expr` is `Alias(inner, name)` AND `name` is in
/// `target_aliases` AND `inner` is a `Function` whose `FunctionExpr`
/// is `FfiPlugin` with `symbol == target_symbol`.
fn expr_matches_target(expr: &Expr, target_aliases: &[String], target_symbol: &str) -> bool {
    let Expr::Alias(inner, alias_name) = expr else {
        return false;
    };
    if !target_aliases.iter().any(|a| a.as_str() == alias_name.as_str()) {
        return false;
    }
    let Expr::Function { function, .. } = inner.as_ref() else {
        return false;
    };
    matches!(
        function,
        FunctionExpr::FfiPlugin { symbol, .. } if symbol.as_str() == target_symbol
    )
}

fn scrub_recursive(plan: &mut DslPlan, target_aliases: &[String], target_symbol: &str) -> usize {
    let mut removed = 0usize;
    match plan {
        DslPlan::HStack { input, exprs, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
            let before = exprs.len();
            exprs.retain(|e| !expr_matches_target(e, target_aliases, target_symbol));
            removed += before - exprs.len();
        }
        DslPlan::Select { input, expr, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
            let before = expr.len();
            expr.retain(|e| !expr_matches_target(e, target_aliases, target_symbol));
            removed += before - expr.len();
        }
        DslPlan::Filter { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::Cache { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::GroupBy { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::Join {
            input_left,
            input_right,
            ..
        } => {
            removed += scrub_recursive(Arc::make_mut(input_left), target_aliases, target_symbol);
            removed += scrub_recursive(Arc::make_mut(input_right), target_aliases, target_symbol);
        }
        DslPlan::MatchToSchema { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::PipeWithSchema { input, .. } => {
            let mut owned: Vec<DslPlan> = input.as_ref().to_vec();
            for child in owned.iter_mut() {
                removed += scrub_recursive(child, target_aliases, target_symbol);
            }
            *input = Arc::from(owned);
        }
        DslPlan::Distinct { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::Sort { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::Slice { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::MapFunction { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::Union { inputs, .. } => {
            for child in inputs.iter_mut() {
                removed += scrub_recursive(child, target_aliases, target_symbol);
            }
        }
        DslPlan::HConcat { inputs, .. } => {
            for child in inputs.iter_mut() {
                removed += scrub_recursive(child, target_aliases, target_symbol);
            }
        }
        DslPlan::ExtContext { input, contexts } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
            for c in contexts.iter_mut() {
                removed += scrub_recursive(c, target_aliases, target_symbol);
            }
        }
        DslPlan::Sink { input, .. } => {
            removed += scrub_recursive(Arc::make_mut(input), target_aliases, target_symbol);
        }
        DslPlan::SinkMultiple { inputs } => {
            for child in inputs.iter_mut() {
                removed += scrub_recursive(child, target_aliases, target_symbol);
            }
        }
        DslPlan::IR { dsl, .. } => {
            removed += scrub_recursive(Arc::make_mut(dsl), target_aliases, target_symbol);
        }
        DslPlan::Scan { .. } | DslPlan::DataFrameScan { .. } => {}
        // Future or cfg-gated variants (PythonScan, MergeSorted, Pivot)
        _ => {}
    }
    removed
}

/// Scrub `(plan_bytes, target_aliases, target_symbol)` and return the
/// rewritten plan bytes plus the count of expressions removed. On
/// `removed == 0`, the returned bytes are byte-equivalent to the
/// input — callers may keep the original `LazyFrame` rather than
/// round-tripping through deserialize+serialize.
pub fn scrub_plugin_expressions(
    plan_bytes: &[u8],
    target_aliases: &[String],
    target_symbol: &str,
) -> Result<(Vec<u8>, usize), String> {
    let mut plan = deserialize_plan(plan_bytes)?;
    let removed = scrub_recursive(&mut plan, target_aliases, target_symbol);
    if removed == 0 {
        return Ok((plan_bytes.to_vec(), 0));
    }
    let new_bytes = serialize_plan(&plan)?;
    Ok((new_bytes, removed))
}
