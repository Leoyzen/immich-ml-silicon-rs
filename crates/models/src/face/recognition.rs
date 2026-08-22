//! ArcFace face recognition — port of immich_ml/models/facial_recognition/recognition.py

use std::path::PathBuf;
use std::sync::Mutex;
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::ep::CoreML;
use ort::ep::coreml::{ModelFormat, SpecializationStrategy, ComputeUnits};
use ort::value::Tensor;


use crate::transforms::{decode_image, normalize};
use crate::face::ops::{align_face, ALIGNED_SIZE};
use crate::{FaceDetectionOutput, DetectedFace, BoundingBox};
use immich_ml_backends::{FaceRecognitionBackend, ImageInput, BackendError};

pub struct FaceRecognizer {
    sessions: Vec<Mutex<Session>>,
}

impl FaceRecognizer {
    /// Load ArcFace recognition model from ONNX file.
    /// Creates `num_sessions` independent sessions for parallel inference.
    pub fn load(model_path: PathBuf, device: &str, num_sessions: usize) -> Result<Self, String> {
        let num_sessions = num_sessions.clamp(1, 4);
        let mut sessions = Vec::with_capacity(num_sessions);

        for i in 0..num_sessions {
            let mut builder = Session::builder()
                .map_err(|e| format!("Session builder: {}", e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("Opt level: {}", e))?;

            if device == "coreml" {
                let ep = CoreML::default()
                    .with_model_format(ModelFormat::MLProgram)
                    .with_compute_units(ComputeUnits::All)
                    .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                    .build();
                builder = builder
                    .with_execution_providers([ep])
                    .map_err(|e| format!("CoreML EP: {}", e))?;
            }

            let session = builder
                .commit_from_file(&model_path)
                .map_err(|e| format!("Load model {:?} (session {}): {}", model_path, i, e))?;
            sessions.push(Mutex::new(session));
        }

        tracing::info!("FaceRecognizer loaded with {} session(s)", num_sessions);

        Ok(Self { sessions })
    }

    /// Run face recognition on detected faces.
    /// Returns one DetectedFace per input face with 512-d embedding.
    pub fn recognize(
        &self,
        image_bytes: &[u8],
        detection: &FaceDetectionOutput,
    ) -> Result<Vec<DetectedFace>, String> {
        if detection.boxes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Decode image
        let img = decode_image(image_bytes)?;
        let rgb = img.to_rgb8();

        // 2. Process each face individually (batch=1) to avoid CoreML dynamic-shape errors
        // Pick least-contended session
        let mut session_idx = 0usize;
        let mut session_guard = None;
        for (i, s) in self.sessions.iter().enumerate() {
            if let Ok(guard) = s.try_lock() {
                session_idx = i;
                session_guard = Some(guard);
                break;
            }
        }
        let mut session = match session_guard {
            Some(g) => g,
            None => {
                session_idx = 0;
                self.sessions[0].lock().unwrap()
            }
        };
        tracing::trace!("FaceRecognizer using session {}", session_idx);

        let mut results = Vec::with_capacity(detection.boxes.len());

        for (i, landmarks) in detection.landmarks.iter().enumerate() {
            // Flatten landmarks to [f32; 10]
            let mut kps = [0.0f32; 10];
            for j in 0..5 {
                kps[j * 2] = landmarks[j][0];
                kps[j * 2 + 1] = landmarks[j][1];
            }
            let aligned = align_face(&rgb, &kps);

            // Convert to CHW float32 [1, 3, 112, 112]
            let mut crop = Array4::<f32>::zeros((1, 3, ALIGNED_SIZE as usize, ALIGNED_SIZE as usize));
            for y in 0..ALIGNED_SIZE as usize {
                for x in 0..ALIGNED_SIZE as usize {
                    let pixel = aligned.get_pixel(x as u32, y as u32);
                    crop[[0, 0, y, x]] = pixel[0] as f32;
                    crop[[0, 1, y, x]] = pixel[1] as f32;
                    crop[[0, 2, y, x]] = pixel[2] as f32;
                }
            }

            // Normalize: mean=127.5, std=127.5
            normalize(crop.as_slice_mut().unwrap(), 127.5, 127.5);

            // ONNX inference (single face)
            let input_tensor = Tensor::from_array(crop)
                .map_err(|e| format!("Tensor create: {}", e))?;
            let outputs = session.run(ort::inputs![input_tensor.view()])
                .map_err(|e| format!("Inference: {}", e))?;

            // Extract embedding [1, 512]
            let emb_view = outputs[0].try_extract_array::<f32>()
                .map_err(|e| format!("Extract embeddings: {}", e))?;
            let emb_dim = emb_view.shape().last().copied().unwrap_or(512);
            let embedding: Vec<f32> = emb_view.iter().copied().take(emb_dim).collect();

            // L2 normalize
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            let normalized: Vec<f32> = embedding.iter().map(|v| v / norm.max(1e-10)).collect();
            
            // Serialize as JSON string (matching orjson.OPT_SERIALIZE_NUMPY)
            let embedding_str = serde_json::to_string(&normalized)
                .map_err(|e| format!("Serialize embedding: {}", e))?;

            // Bounding box as integers
            let box_ = &detection.boxes[i];
            let bounding_box = BoundingBox {
                x1: box_[0].round() as i32,
                y1: box_[1].round() as i32,
                x2: box_[2].round() as i32,
                y2: box_[3].round() as i32,
            };

            results.push(DetectedFace {
                bounding_box,
                embedding: embedding_str,
                score: detection.scores[i],
            });
        }

        Ok(results)
    }
}

#[async_trait::async_trait]
impl FaceRecognitionBackend for FaceRecognizer {
    async fn recognize(&self, image: &ImageInput, detection: &FaceDetectionOutput) -> Result<Vec<DetectedFace>, BackendError> {
        FaceRecognizer::recognize(self, &image.bytes, detection).map_err(BackendError::Other)
    }
}
