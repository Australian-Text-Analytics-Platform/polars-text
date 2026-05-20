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

fn collect_scan_sources(sources: &ScanSources, out: &mut Vec<String>) {
    if let ScanSources::Paths(paths) = sources {
        for p in paths.as_ref() {
            out.push(p.as_str().to_string());
        }
    }
}

fn rewrite_scan_sources(sources: &mut ScanSources, mapper: &HashMap<String, String>) -> usize {
    let ScanSources::Paths(paths) = sources else {
        return 0;
    };

    let mut changed = 0usize;
    let mut new_paths: Vec<PlRefPath> = Vec::with_capacity(paths.len());
    for p in paths.as_ref() {
        let current = p.as_str();
        if let Some(replacement) = mapper.get(current) {
            new_paths.push(PlRefPath::new(replacement.as_str()));
            changed += 1;
        } else {
            new_paths.push(p.clone());
        }
    }

    if changed > 0 {
        *sources = ScanSources::Paths(Buffer::from_iter(new_paths));
    }

    changed
}

fn visit_children_mut(plan: &mut DslPlan, visit: &mut impl FnMut(&mut DslPlan)) {
    match plan {
        DslPlan::Filter { input, .. }
        | DslPlan::Cache { input, .. }
        | DslPlan::Select { input, .. }
        | DslPlan::GroupBy { input, .. }
        | DslPlan::HStack { input, .. }
        | DslPlan::MatchToSchema { input, .. }
        | DslPlan::Distinct { input, .. }
        | DslPlan::Sort { input, .. }
        | DslPlan::Slice { input, .. }
        | DslPlan::MapFunction { input, .. }
        | DslPlan::Sink { input, .. } => visit(Arc::make_mut(input)),
        DslPlan::Join {
            input_left,
            input_right,
            ..
        } => {
            visit(Arc::make_mut(input_left));
            visit(Arc::make_mut(input_right));
        }
        DslPlan::PipeWithSchema { input, .. } => {
            let mut owned: Vec<DslPlan> = input.as_ref().to_vec();
            for child in owned.iter_mut() {
                visit(child);
            }
            *input = Arc::from(owned);
        }
        DslPlan::Union { inputs, .. } | DslPlan::HConcat { inputs, .. } => {
            for child in inputs.iter_mut() {
                visit(child);
            }
        }
        DslPlan::ExtContext { input, contexts } => {
            visit(Arc::make_mut(input));
            for context in contexts.iter_mut() {
                visit(context);
            }
        }
        DslPlan::SinkMultiple { inputs } => {
            for child in inputs.iter_mut() {
                visit(child);
            }
        }
        DslPlan::IR { dsl, .. } => visit(Arc::make_mut(dsl)),
        DslPlan::Scan { .. } | DslPlan::DataFrameScan { .. } => {}
        _ => {}
    }
}

fn walk_rewrite(plan: &mut DslPlan, mapper: &HashMap<String, String>) -> usize {
    let mut changed = match plan {
        DslPlan::Scan { sources, .. } => rewrite_scan_sources(sources, mapper),
        _ => 0,
    };

    visit_children_mut(plan, &mut |child| {
        changed += walk_rewrite(child, mapper);
    });

    changed
}

pub fn list_source_paths(path: &Path) -> Result<Vec<String>, String> {
    let plan = load_plan(path)?;
    let mut out = Vec::new();
    for node in &plan {
        if let DslPlan::Scan { sources, .. } = node {
            collect_scan_sources(sources, &mut out);
        }
    }
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
