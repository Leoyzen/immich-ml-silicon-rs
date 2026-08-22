//! Face detection (SCRFD) and recognition (ArcFace) via local ONNX inference.
//! Cloud CLIP/OCR handled by the `immich-ml-cloud` crate.

pub mod transforms;
pub mod face;
pub mod download;

pub use face::detection::FaceDetector;
pub use face::recognition::FaceRecognizer;
pub use face::ops::{DET_SIZE, ALIGNED_SIZE};

// Re-export shared types from the backends crate.
pub use immich_ml_backends::{
    FaceDetectionOutput, DetectedFace, BoundingBox, ImageInput, BackendError,
};
