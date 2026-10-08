//! Progress reporting through a small JSON file (Wordflow issue 350).
//!
//! A Polars expression plugin cannot call back into Python, so long-running
//! expressions (topic modelling, tokenising a large corpus) report progress by
//! rewriting one JSON file that the caller polls:
//!
//! ```json
//! {"step": 2, "steps": 5, "label": "embedding", "done": 18900,
//!  "total": 114461, "unit": "segments", "updated_at_ms": 1791428969000}
//! ```
//!
//! Writes are throttled (at most every `MIN_WRITE_INTERVAL`, plus every step
//! change) and atomic (a temporary file renamed over the target), so a reader
//! never sees half a file. Every expression call given the same path shares
//! one counter, because Polars may run an elementwise expression on several
//! chunks of a column at once; their counts add up. `total` is `None` when the
//! expression cannot know it (the tokeniser sees only its own chunk).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MIN_WRITE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(serde::Serialize, Clone, Debug, PartialEq)]
struct Snapshot {
    step: u32,
    steps: u32,
    label: String,
    done: u64,
    total: Option<u64>,
    unit: String,
    updated_at_ms: u64,
}

struct State {
    snapshot: Snapshot,
    last_write: Option<Instant>,
}

/// One progress file, shared by every expression call that names its path.
pub struct Progress {
    path: PathBuf,
    state: Mutex<State>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<Progress>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<Progress>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// The shared progress for `path`, or `None` when no path was given.
pub fn for_path(path: Option<&str>) -> Option<Arc<Progress>> {
    let path = PathBuf::from(path.filter(|value| !value.is_empty())?);
    let mut entries = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    Some(Arc::clone(entries.entry(path.clone()).or_insert_with(
        || {
            Arc::new(Progress {
                path,
                state: Mutex::new(State {
                    snapshot: Snapshot {
                        step: 1,
                        steps: 1,
                        label: String::new(),
                        done: 0,
                        total: None,
                        unit: String::new(),
                        updated_at_ms: now_ms(),
                    },
                    last_write: None,
                }),
            })
        },
    )))
}

impl Progress {
    /// Start step `step` of `steps`, written at once so the reader sees it.
    #[cfg(feature = "topic-modeling")]
    pub fn start_step(&self, step: u32, steps: u32, label: &str, total: Option<u64>, unit: &str) {
        self.update(true, |snapshot| {
            snapshot.step = step;
            snapshot.steps = steps;
            snapshot.label = label.to_owned();
            snapshot.done = 0;
            snapshot.total = total;
            snapshot.unit = unit.to_owned();
        });
    }

    /// Join a step that other calls may already have started: only the first
    /// call sets it, so the shared count is never reset (chunks of one column).
    pub fn join_step(&self, step: u32, steps: u32, label: &str, total: Option<u64>, unit: &str) {
        let started = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            !state.snapshot.label.is_empty()
        };
        if !started {
            self.update(true, |snapshot| {
                if snapshot.label.is_empty() {
                    snapshot.step = step;
                    snapshot.steps = steps;
                    snapshot.label = label.to_owned();
                    snapshot.total = total;
                    snapshot.unit = unit.to_owned();
                }
            });
        }
    }

    /// Add `count` finished items to the current step.
    pub fn add(&self, count: u64) {
        self.update(false, |snapshot| snapshot.done += count);
    }

    /// Set how many items of the current step are finished.
    #[cfg(feature = "topic-modeling")]
    pub fn set_done(&self, done: u64) {
        self.update(false, |snapshot| snapshot.done = done);
    }

    /// Write the current state now (for example when a step ends).
    pub fn flush(&self) {
        self.update(true, |_| {});
    }

    fn update(&self, force: bool, change: impl FnOnce(&mut Snapshot)) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        change(&mut state.snapshot);
        let due = state
            .last_write
            .is_none_or(|last| last.elapsed() >= MIN_WRITE_INTERVAL);
        if !(force || due) {
            return;
        }
        state.snapshot.updated_at_ms = now_ms();
        // Progress is advisory: a failed write must never fail the analysis.
        if self.write(&state.snapshot).is_ok() {
            state.last_write = Some(Instant::now());
        }
    }

    fn write(&self, snapshot: &Snapshot) -> std::io::Result<()> {
        let mut temporary = self.path.clone().into_os_string();
        temporary.push(format!(".{}.tmp", std::process::id()));
        let temporary = PathBuf::from(temporary);
        std::fs::write(&temporary, serde_json::to_vec(snapshot)?)?;
        std::fs::rename(&temporary, &self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn no_path_means_no_progress() {
        assert!(for_path(None).is_none());
        assert!(for_path(Some("")).is_none());
    }

    #[cfg(feature = "topic-modeling")]
    #[test]
    fn steps_are_written_at_once_and_counts_are_throttled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress.json");
        let progress = for_path(path.to_str()).unwrap();

        progress.start_step(2, 5, "embedding", Some(100), "segments");
        let first = read(&path);
        assert_eq!(first["step"], 2);
        assert_eq!(first["steps"], 5);
        assert_eq!(first["label"], "embedding");
        assert_eq!(first["done"], 0);
        assert_eq!(first["total"], 100);

        // Within the throttle window the file is not rewritten...
        progress.add(10);
        assert_eq!(read(&path)["done"], 0);
        // ...but a flush writes the count so far.
        progress.flush();
        assert_eq!(read(&path)["done"], 10);
        assert!(!dir
            .path()
            .join(format!("progress.json.{}.tmp", std::process::id()))
            .exists());
    }

    #[test]
    fn calls_sharing_a_path_add_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.json");
        let first = for_path(path.to_str()).unwrap();
        let second = for_path(path.to_str()).unwrap();
        first.join_step(1, 1, "tokenizing", None, "documents");
        // A second chunk joining later must not reset the shared count.
        first.add(1);
        second.join_step(1, 1, "tokenizing", None, "documents");
        std::thread::scope(|scope| {
            scope.spawn(|| (0..50).for_each(|_| first.add(1)));
            scope.spawn(|| (0..50).for_each(|_| second.add(1)));
        });
        first.flush();
        let written = read(&path);
        assert_eq!(written["done"], 101);
        assert!(written["total"].is_null());
    }
}
