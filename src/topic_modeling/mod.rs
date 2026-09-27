//! Polars result serialization; computation lives in ldaca-rs.
pub mod plugin;
pub use ldaca_rs::topic_modeling::{projection, run, segmentation, RunConfig, TopicModelingResult};
