use std::collections::HashMap;
use serde::Deserialize;

/// Entry in the pipeline request: {modelName, options}
#[derive(Debug, Deserialize)]
pub struct PipelineEntry {
    #[serde(rename = "modelName")]
    pub model_name: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// Top-level pipeline request: {task: {type: {modelName, options}}}
pub type PipelineRequest = HashMap<String, HashMap<String, PipelineEntry>>;

/// Flattened inference entry for dispatch.
#[derive(Debug, Clone)]
pub struct InferenceEntry {
    pub task: String,       // "facial-recognition", "clip", "ocr"
    pub r#type: String,     // "detection", "recognition", "visual", "textual"
    pub model_name: String,
    pub options: HashMap<String, serde_json::Value>,
}

/// Payload for inference — either image bytes or text.
pub enum Payload {
    Image(Vec<u8>),
    Text(String),
}
