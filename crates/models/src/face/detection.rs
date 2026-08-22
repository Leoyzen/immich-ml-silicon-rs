//! SCRFD face detection — port of immich_ml/models/facial_recognition/detection.py

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::ep::CoreML;
use ort::ep::coreml::{ModelFormat, SpecializationStrategy, ComputeUnits};
use ort::value::Tensor;

use crate::transforms::{decode_image, letterbox, normalize};
use crate::face::ops::{decode_scrfd, nms, DET_SIZE};
use crate::FaceDetectionOutput;
use immich_ml_backends::{FaceDetectionBackend, ImageInput, BackendError};

pub struct FaceDetector {
    sessions: Arc<Vec<Mutex<Session>>>,
}

impl FaceDetector {
    /// Load SCRFD detection model from ONNX file.
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

        tracing::info!("FaceDetector loaded with {} session(s)", num_sessions);

        Ok(Self { sessions: Arc::new(sessions) })
    }

    /// Run face detection on image (sync, CPU-bound).
    /// Accepts ImageInput so pre-decoded RGBA can be reused if available.
    /// Returns boxes [x1,y1,x2,y2], scores, and 5 landmarks per face.
    fn detect_with_sessions(
        sessions: &[Mutex<Session>],
        image: &ImageInput,
        min_score: f32,
    ) -> Result<FaceDetectionOutput, String> {
        // 1. Decode image (use pre-decoded RGBA if available)
        let decoded_fallback;
        let (rgba, w, h) = if let Some(ref d) = image.decoded {
            (d.rgba.as_slice(), d.width, d.height)
        } else {
            decoded_fallback = decode_image(&image.bytes)?;
            (decoded_fallback.0.as_slice(), decoded_fallback.1, decoded_fallback.2)
        };

        // 2. Letterbox to 640×640
        let (canvas, scale) = letterbox((rgba, w, h), DET_SIZE);

        // 3. Convert to NCHW float32, normalize: mean=127.5, std=128
        // Use bulk pixel access via into_raw() instead of per-pixel get_pixel().
        let det_size = DET_SIZE as usize;
        let mut input_data = Array4::<f32>::zeros((1, 3, det_size, det_size));
        let raw = canvas.into_raw(); // Vec<u8>, RGB order, 3 bytes per pixel
        for (i, chunk) in raw.chunks_exact(3).enumerate() {
            let y = i / det_size;
            let x = i % det_size;
            input_data[[0, 0, y, x]] = chunk[0] as f32;
            input_data[[0, 1, y, x]] = chunk[1] as f32;
            input_data[[0, 2, y, x]] = chunk[2] as f32;
        }
        normalize(input_data.as_slice_mut().unwrap(), 127.5, 128.0);

        // 4. ONNX inference — pick least-contended session
        let input_tensor = Tensor::from_array(input_data)
            .map_err(|e| format!("Tensor create: {}", e))?;

        let mut session_idx = 0usize;
        let mut session_guard = None;
        for (i, s) in sessions.iter().enumerate() {
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
                sessions[0].lock().unwrap()
            }
        };
        tracing::trace!("FaceDetector using session {}", session_idx);

        let outputs = session.run(ort::inputs![input_tensor.view()])
            .map_err(|e| format!("Inference: {}", e))?;

        // 5. Extract output heads
        let mut heads_data = Vec::new();
        let mut heads_shapes = Vec::new();
        for i in 0..outputs.len() {
            let output = &outputs[i];
            let view = output.try_extract_array::<f32>()
                .map_err(|e| format!("Extract output {}: {}", i, e))?;
            let shape: Vec<usize> = view.shape().to_vec();
            heads_data.push(view.iter().copied().collect::<Vec<f32>>());
            heads_shapes.push((
                shape.get(1).copied().unwrap_or(0),
                shape.get(2).copied().unwrap_or(0),
                shape.get(3).copied().unwrap_or(0),
            ));
        }

        // 6. Decode SCRFD predictions
        let (scores, boxes, kps) = decode_scrfd(&heads_data, &heads_shapes);

        // 7. Filter by min_score
        let mut filtered_boxes = Vec::new();
        let mut filtered_scores = Vec::new();
        let mut filtered_kps = Vec::new();
        for i in 0..scores.len() {
            if scores[i] >= min_score {
                // Scale back to original image coordinates
                let inv_scale = 1.0 / scale;
                filtered_boxes.push([
                    boxes[i][0] * inv_scale,
                    boxes[i][1] * inv_scale,
                    boxes[i][2] * inv_scale,
                    boxes[i][3] * inv_scale,
                ]);
                filtered_scores.push(scores[i]);
                let mut kp = [0.0f32; 10];
                for j in 0..5 {
                    kp[j * 2] = kps[i][j * 2] * inv_scale;
                    kp[j * 2 + 1] = kps[i][j * 2 + 1] * inv_scale;
                }
                filtered_kps.push(kp);
            }
        }

        // 8. NMS
        let keep = nms(&filtered_boxes, &filtered_scores, 0.4);

        // 9. Build output with landmarks as [[f32;2]; 5]
        let mut result = FaceDetectionOutput::default();
        for &idx in &keep {
            result.boxes.push(filtered_boxes[idx]);
            result.scores.push(filtered_scores[idx]);
            let mut landmarks = [[0.0f32; 2]; 5];
            for j in 0..5 {
                landmarks[j] = [filtered_kps[idx][j * 2], filtered_kps[idx][j * 2 + 1]];
            }
            result.landmarks.push(landmarks);
        }

        Ok(result)
    }
}

#[async_trait::async_trait]
impl FaceDetectionBackend for FaceDetector {
    async fn detect(&self, image: &ImageInput, min_score: f32) -> Result<FaceDetectionOutput, BackendError> {
        let sessions = self.sessions.clone();
        let image = image.clone();
        tokio::task::spawn_blocking(move || {
            FaceDetector::detect_with_sessions(&sessions, &image, min_score)
        })
        .await
        .map_err(|e| BackendError::Other(format!("Join error: {}", e)))?
        .map_err(BackendError::Other)
    }
}
