//! ArcFace face recognition — port of immich_ml/models/facial_recognition/recognition.py

use std::path::PathBuf;
use std::sync::Mutex;
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::ep::CoreML;
use ort::ep::coreml::{ModelFormat, SpecializationStrategy, ComputeUnits};
use ort::value::Tensor;


use crate::transforms::{decode_image, normalize, rgba_to_rgb_image};
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
    /// Accepts ImageInput so pre-decoded RGBA can be reused if available.
    /// Returns one DetectedFace per input face with 512-d embedding.
    pub fn recognize(
        &self,
        image: &ImageInput,
        detection: &FaceDetectionOutput,
    ) -> Result<Vec<DetectedFace>, String> {
        if detection.boxes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Decode image (use pre-decoded RGBA if available)
        let decoded_fallback;
        let (rgba, w, h) = if let Some(ref d) = image.decoded {
            (d.rgba.as_slice(), d.width, d.height)
        } else {
            decoded_fallback = decode_image(&image.bytes)?;
            (decoded_fallback.0.as_slice(), decoded_fallback.1, decoded_fallback.2)
        };
        let rgb = rgba_to_rgb_image(rgba, w, h);

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
            // Use bulk pixel access via into_raw() instead of per-pixel get_pixel().
            let aligned_size = ALIGNED_SIZE as usize;
            let mut crop = Array4::<f32>::zeros((1, 3, aligned_size, aligned_size));
            let raw = aligned.into_raw(); // Vec<u8>, RGB order, 3 bytes per pixel
            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                let y = i / aligned_size;
                let x = i % aligned_size;
                crop[[0, 0, y, x]] = chunk[0] as f32;
                crop[[0, 1, y, x]] = chunk[1] as f32;
                crop[[0, 2, y, x]] = chunk[2] as f32;
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
        FaceRecognizer::recognize(self, image, detection).map_err(BackendError::Other)
    }
}
