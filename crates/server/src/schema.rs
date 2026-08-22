use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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

/// Bounding box with integer coordinates (matching immich-ml BoundingBox TypedDict).
#[derive(Debug, Serialize)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Single detected face with embedding.
#[derive(Debug, Serialize)]
pub struct DetectedFace {
    #[serde(rename = "boundingBox")]
    pub bounding_box: BoundingBox,
    pub embedding: String, // JSON-serialized float array
    pub score: f32,
}

/// OCR result matching immich's asset-ocr schema.
/// Box coordinates are normalized to [0,1].
#[derive(Debug, Serialize, Default)]
pub struct OcrResult {
    pub text: Vec<String>,
    pub box_: Vec<f64>,     // flattened: [x1,y1,x2,y1,x2,y2,x1,y2, ...]
    #[serde(rename = "boxScore")]
    pub box_score: Vec<f64>,
    #[serde(rename = "textScore")]
    pub text_score: Vec<f64>,
}

/// Intermediate face detection output (before recognition).
#[derive(Debug, Clone)]
pub struct FaceDetectionOutput {
    pub boxes: Vec<[f32; 4]>,     // [x1, y1, x2, y2] in original image coords
    pub scores: Vec<f32>,
    pub landmarks: Vec<[[f32; 2]; 5]>, // 5 keypoints per face
}

/// Payload for inference — either image bytes or text.
pub enum Payload {
    Image(Vec<u8>),
    Text(String),
}
