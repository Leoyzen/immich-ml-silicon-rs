//! Trait abstractions for ML backends.
//! Supports pluggable implementations: ONNX (local), DashScope (cloud), Apple Vision (native).

use async_trait::async_trait;

// ── Shared types ──────────────────────────────────────────────

/// Pre-decoded RGBA image data (avoids re-decoding for each pipeline stage).
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Image input with decoded dimensions (pre-preprocessing).
#[derive(Debug, Clone)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Pre-decoded RGBA pixels, if available, to avoid redundant decoding.
    pub decoded: Option<DecodedImage>,
}

/// Unified error type for all backends.
#[derive(Debug)]
pub enum BackendError {
    /// Backend-specific error message (model load, inference, image decode, etc.)
    Other(String),
    /// API error (status code + message) — for cloud backends.
    Api { status: u16, message: String },
    /// Network error — for cloud backends.
    Network(String),
}

impl BackendError {
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Api { status, .. } => *status == 429 || *status >= 500,
            Self::Network(_) => true,
            Self::Other(_) => false,
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other(msg) => write!(f, "{}", msg),
            Self::Api { status, message } => write!(f, "API {}: {}", status, message),
            Self::Network(msg) => write!(f, "Network: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Intermediate face detection output (before recognition).
#[derive(Debug, Clone, Default)]
pub struct FaceDetectionOutput {
    pub boxes: Vec<[f32; 4]>,          // [x1, y1, x2, y2] in original image coords
    pub scores: Vec<f32>,
    pub landmarks: Vec<[[f32; 2]; 5]>, // 5 keypoints per face
}

/// Single detected face with embedding (matches immich schema).
#[derive(Debug, serde::Serialize)]
pub struct DetectedFace {
    #[serde(rename = "boundingBox")]
    pub bounding_box: BoundingBox,
    pub embedding: String,
    pub score: f32,
}

#[derive(Debug, serde::Serialize)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// OCR result matching immich's asset-ocr schema.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct OcrResult {
    pub text: Vec<String>,
    #[serde(rename = "box")]
    pub box_coords: Vec<f64>, // flattened: [x1,y1,x2,y1,x2,y2,x1,y2, ...] normalized [0,1]
    #[serde(rename = "boxScore")]
    pub box_score: Vec<f64>,
    #[serde(rename = "textScore")]
    pub text_score: Vec<f64>,
}

// ── Traits ────────────────────────────────────────────────────

#[async_trait]
pub trait FaceDetectionBackend: Send + Sync {
    /// Detect faces in an image. Returns boxes, scores, and landmarks.
    async fn detect(
        &self,
        image: &ImageInput,
        min_score: f32,
    ) -> Result<FaceDetectionOutput, BackendError>;
}

#[async_trait]
pub trait FaceRecognitionBackend: Send + Sync {
    /// Recognize faces: takes detection output, returns embeddings.
    async fn recognize(
        &self,
        image: &ImageInput,
        detection: &FaceDetectionOutput,
    ) -> Result<Vec<DetectedFace>, BackendError>;
}

#[async_trait]
pub trait ClipBackend: Send + Sync {
    /// Encode an image into a vector embedding.
    async fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, BackendError>;
    /// Encode text into a vector embedding.
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>, BackendError>;

    /// Encode multiple images in a single batch call.
    /// Returns embeddings in the same order as the input images.
    /// Default implementation calls `encode_image` one by one.
    async fn encode_image_batch(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>, BackendError> {
        let mut results = Vec::with_capacity(images.len());
        for img in images {
            results.push(self.encode_image(img).await?);
        }
        Ok(results)
    }
}

#[async_trait]
pub trait OcrBackend: Send + Sync {
    /// Extract text from an image.
    async fn recognize(&self, image_bytes: &[u8]) -> Result<OcrResult, BackendError>;
    /// Whether this backend returns real bounding boxes (vs synthesized).
    fn has_bounding_boxes(&self) -> bool {
        false
    }
}
