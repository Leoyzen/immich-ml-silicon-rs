use std::sync::Arc;
use crate::config::Config;
use crate::concurrency::ConcurrencyControl;
use immich_ml_backends::{FaceDetectionBackend, FaceRecognitionBackend, ClipBackend, OcrBackend};

#[cfg(target_os = "macos")]
use immich_ml_vision::{VisionOcrBackend, VisionFaceDetector};

pub struct AppState {
    pub config: Config,
    pub concurrency: ConcurrencyControl,
    pub face_detector: Arc<dyn FaceDetectionBackend>,
    pub face_recognizer: Arc<dyn FaceRecognitionBackend>,
    pub clip: Arc<dyn ClipBackend>,
    pub ocr: Arc<dyn OcrBackend>,
}

impl AppState {
    pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let concurrency = ConcurrencyControl::new(config.max_concurrency);

        // Face detection backend
        let face_detector: Arc<dyn FaceDetectionBackend> = if config.face_detection_backend == "onnx" {
            Arc::new(immich_ml_models::FaceDetector::load(
                config.det_model_path.clone(),
                &config.device,
                config.onnx_sessions,
            ).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?)
        } else if config.face_detection_backend == "vision" {
            #[cfg(target_os = "macos")]
            {
                Arc::new(VisionFaceDetector::new())
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("Vision backend requires macOS".into());
            }
        } else {
            return Err(format!("Unknown face_detection_backend: {}", config.face_detection_backend).into());
        };

        // Face recognition backend
        let face_recognizer: Arc<dyn FaceRecognitionBackend> = if config.face_recognition_backend == "onnx" {
            Arc::new(immich_ml_models::FaceRecognizer::load(
                config.rec_model_path.clone(),
                &config.device,
                config.onnx_sessions,
            ).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?)
        } else {
            return Err(format!("Unknown face_recognition_backend: {}", config.face_recognition_backend).into());
        };

        // CLIP + OCR backends.
        // When both use DashScope, share a single DashScopeClient instance
        // (cloning is cheap — reqwest::Client shares its connection pool).
        let dashscope_client = if config.clip_backend == "dashscope" || config.ocr_backend == "dashscope" {
            Some(immich_ml_cloud::DashScopeClient::new(
                config.dashscope_api_key.clone(),
                config.clip_model.clone(),
                config.ocr_model.clone(),
                config.clip_dim,
            ))
        } else {
            None
        };

        let clip: Arc<dyn ClipBackend> = if config.clip_backend == "dashscope" {
            Arc::new(dashscope_client.as_ref().unwrap().clone())
        } else {
            return Err(format!("Unknown clip_backend: {}", config.clip_backend).into());
        };

        let ocr: Arc<dyn OcrBackend> = if config.ocr_backend == "dashscope" {
            // Reuse the same DashScopeClient — cloning shares the reqwest
            // connection pool, avoiding duplicate connection overhead.
            Arc::new(dashscope_client.as_ref().unwrap().clone())
        } else if config.ocr_backend == "vision" {
            #[cfg(target_os = "macos")]
            {
                Arc::new(VisionOcrBackend::new())
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("Vision backend requires macOS".into());
            }
        } else {
            return Err(format!("Unknown ocr_backend: {}", config.ocr_backend).into());
        };

        Ok(Self {
            config,
            concurrency,
            face_detector,
            face_recognizer,
            clip,
            ocr,
        })
    }
}
