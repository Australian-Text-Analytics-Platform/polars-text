//! Inspect and rewrite scan source paths inside a serialized Polars
//! `LazyFrame` plan (`.plbin` file).
//!
//! Used by the backend to migrate workspace plans across user home
//! directories: plan files are portable, the backend computes the
//! mapping between the old parent and the new parent, and this helper
//! rewrites the absolute paths embedded in the DSL.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_plan::dsl::{DslPlan, PlanSerializationContext, ScanSources};
use polars_utils::pl_path::PlRefPath;

fn load_plan(path: &Path) -> Result<DslPlan, String> {
    let file = File::open(path).map_err(|e| format!("failed to open plan file: {e}"))?;
    let reader = BufReader::new(file);
    DslPlan::deserialize_versioned(reader).map_err(|e| format!("failed to deserialize plan: {e}"))
}

fn store_plan(path: &Path, plan: &DslPlan) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("failed to create plan file: {e}"))?;
    let writer = BufWriter::new(file);
    plan.serialize_versioned(writer, PlanSerializationContext::default())
        .map_err(|e| format!("failed to serialize plan: {e}"))
}

fn walk_collect(plan: &DslPlan, out: &mut Vec<String>) {
    match plan {
        DslPlan::Scan { sources, .. } => {
            if let ScanSources::Paths(paths) = sources {
                for p in paths.as_ref() {
                    out.push(p.as_str().to_string());
                }
            }
        }
        DslPlan::Filter { input, .. } => walk_collect(input, out),
        DslPlan::Cache { input, .. } => walk_collect(input, out),
        DslPlan::Select { input, .. } => walk_collect(input, out),
        DslPlan::GroupBy { input, .. } => walk_collect(input, out),
        DslPlan::Join {
            input_left,
            input_right,
            ..
        } => {
            walk_collect(input_left, out);
            walk_collect(input_right, out);
        }
        DslPlan::HStack { input, .. } => walk_collect(input, out),
        DslPlan::MatchToSchema { input, .. } => walk_collect(input, out),
        DslPlan::PipeWithSchema { input, .. } => {
            for child in input.iter() {
                walk_collect(child, out);
            }
        }
        DslPlan::Distinct { input, .. } => walk_collect(input, out),
        DslPlan::Sort { input, .. } => walk_collect(input, out),
        DslPlan::Slice { input, .. } => walk_collect(input, out),
        DslPlan::MapFunction { input, .. } => walk_collect(input, out),
        DslPlan::Union { inputs, .. } => {
            for child in inputs {
                walk_collect(child, out);
            }
        }
        DslPlan::HConcat { inputs, .. } => {
            for child in inputs {
                walk_collect(child, out);
            }
        }
        DslPlan::ExtContext { input, contexts } => {
            walk_collect(input, out);
            for c in contexts {
                walk_collect(c, out);
            }
        }
        DslPlan::Sink { input, .. } => walk_collect(input, out),
        DslPlan::SinkMultiple { inputs } => {
            for child in inputs {
                walk_collect(child, out);
            }
        }
        DslPlan::IR { dsl, .. } => walk_collect(dsl, out),
        DslPlan::DataFrameScan { .. } => {}
        // Future or cfg-gated variants (PythonScan, MergeSorted, Pivot)
        _ => {}
    }
}

fn walk_rewrite(plan: &mut DslPlan, mapper: &HashMap<String, String>) -> usize {
    let mut changed = 0usize;
    match plan {
        DslPlan::Scan { sources, .. } => {
            if let ScanSources::Paths(paths) = sources {
                let mut new_paths: Vec<PlRefPath> = Vec::with_capacity(paths.len());
                let mut any_changed = false;
                for p in paths.as_ref() {
                    let current = p.as_str();
                    if let Some(replacement) = mapper.get(current) {
                        new_paths.push(PlRefPath::new(replacement.as_str()));
                        any_changed = true;
                        changed += 1;
                    } else {
                        new_paths.push(p.clone());
                    }
                }
                if any_changed {
                    *sources = ScanSources::Paths(Buffer::from_iter(new_paths));
                }
            }
        }
        DslPlan::Filter { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Cache { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Select { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::GroupBy { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Join {
            input_left,
            input_right,
            ..
        } => {
            changed += walk_rewrite(Arc::make_mut(input_left), mapper);
            changed += walk_rewrite(Arc::make_mut(input_right), mapper);
        }
        DslPlan::HStack { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::MatchToSchema { input, .. } => {
            changed += walk_rewrite(Arc::make_mut(input), mapper)
        }
        DslPlan::PipeWithSchema { input, .. } => {
            let mut owned: Vec<DslPlan> = input.as_ref().to_vec();
            for child in owned.iter_mut() {
                changed += walk_rewrite(child, mapper);
            }
            *input = Arc::from(owned);
        }
        DslPlan::Distinct { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Sort { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Slice { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::MapFunction { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::Union { inputs, .. } => {
            for child in inputs.iter_mut() {
                changed += walk_rewrite(child, mapper);
            }
        }
        DslPlan::HConcat { inputs, .. } => {
            for child in inputs.iter_mut() {
                changed += walk_rewrite(child, mapper);
            }
        }
        DslPlan::ExtContext { input, contexts } => {
            changed += walk_rewrite(Arc::make_mut(input), mapper);
            for c in contexts.iter_mut() {
                changed += walk_rewrite(c, mapper);
            }
        }
        DslPlan::Sink { input, .. } => changed += walk_rewrite(Arc::make_mut(input), mapper),
        DslPlan::SinkMultiple { inputs } => {
            for child in inputs.iter_mut() {
                changed += walk_rewrite(child, mapper);
            }
        }
        DslPlan::IR { dsl, .. } => changed += walk_rewrite(Arc::make_mut(dsl), mapper),
        DslPlan::DataFrameScan { .. } => {}
        _ => {}
    }
    changed
}

pub fn list_source_paths(path: &Path) -> Result<Vec<String>, String> {
    let plan = load_plan(path)?;
    let mut out = Vec::new();
    walk_collect(&plan, &mut out);
    Ok(out)
}

pub fn replace_source_paths(
    path: &Path,
    mapper: &HashMap<String, String>,
) -> Result<usize, String> {
    let mut plan = load_plan(path)?;
    let changed = walk_rewrite(&mut plan, mapper);
    if changed > 0 {
        store_plan(path, &plan)?;
    }
    Ok(changed)
}
