use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub cache_dir: PathBuf,
    pub dashscope_api_key: String,
    pub clip_model: String,
    pub ocr_model: String,
    pub clip_dim: usize,
    pub device: String,
    pub max_concurrency: usize,
    pub face_detection_backend: String,
    pub face_recognition_backend: String,
    pub clip_backend: String,
    pub ocr_backend: String,
    pub det_model_path: PathBuf,
    pub rec_model_path: PathBuf,
    pub onnx_sessions: usize,
    pub clip_batch_size: usize,
    pub clip_batch_interval_ms: u64,
    pub ocr_min_confidence: f32,
}

impl Config {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let port = std::env::var("IMMICH_ML_PORT")
            .unwrap_or_else(|_| "3003".to_string())
            .parse()?;

        let cache_dir = std::env::var("IMMICH_ML_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./model-cache"));

        let dashscope_api_key = std::env::var("DASHSCOPE_API_KEY")
            .map_err(|_| "DASHSCOPE_API_KEY is required")?;

        let clip_model = std::env::var("IMMICH_ML_CLIP_MODEL")
            .unwrap_or_else(|_| "qwen3-vl-embedding".to_string());

        let clip_dim = std::env::var("IMMICH_ML_CLIP_DIM")
            .unwrap_or_else(|_| "512".to_string())
            .parse()?;

        let ocr_model = std::env::var("IMMICH_ML_OCR_MODEL")
            .unwrap_or_else(|_| "qwen-vl-ocr".to_string());

        let device = std::env::var("IMMICH_ML_DEVICE")
            .unwrap_or_else(|_| "coreml".to_string());

        let max_concurrency = std::env::var("IMMICH_ML_MAX_CONCURRENCY")
            .unwrap_or_else(|_| "5".to_string())
            .parse()?;

        let face_detection_backend = std::env::var("IMMICH_ML_FACE_DETECTION_BACKEND")
            .unwrap_or_else(|_| "onnx".to_string());

        let face_recognition_backend = std::env::var("IMMICH_ML_FACE_RECOGNITION_BACKEND")
            .unwrap_or_else(|_| "onnx".to_string());

        let clip_backend = std::env::var("IMMICH_ML_CLIP_BACKEND")
            .unwrap_or_else(|_| "dashscope".to_string());

        let ocr_backend = std::env::var("IMMICH_ML_OCR_BACKEND")
            .unwrap_or_else(|_| "dashscope".to_string());

        let det_model_path = std::env::var("IMMICH_ML_DET_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| cache_dir.join("det_10g.onnx"));

        let rec_model_path = std::env::var("IMMICH_ML_REC_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| cache_dir.join("w600k_mbf23.onnx"));

        let onnx_sessions = std::env::var("IMMICH_ML_ONNX_SESSIONS")
            .unwrap_or_else(|_| "2".to_string())
            .parse::<usize>()
            .unwrap_or(2)
            .clamp(1, 4);

        let clip_batch_size = std::env::var("IMMICH_ML_CLIP_BATCH_SIZE")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<usize>()
            .unwrap_or(10)
            .clamp(1, 10);

        let clip_batch_interval_ms = std::env::var("IMMICH_ML_CLIP_BATCH_INTERVAL_MS")
            .unwrap_or_else(|_| "50".to_string())
            .parse::<u64>()
            .unwrap_or(50);

        let ocr_min_confidence = std::env::var("IMMICH_ML_OCR_MIN_CONFIDENCE")
            .unwrap_or_else(|_| "0.1".to_string())
            .parse::<f32>()
            .unwrap_or(0.1);

        Ok(Self {
            port,
            cache_dir,
            dashscope_api_key,
            clip_model,
            ocr_model,
            clip_dim,
            device,
            max_concurrency,
            face_detection_backend,
            face_recognition_backend,
            clip_backend,
            ocr_backend,
            det_model_path,
            rec_model_path,
            onnx_sessions,
            clip_batch_size,
            clip_batch_interval_ms,
            ocr_min_confidence,
        })
    }
}
