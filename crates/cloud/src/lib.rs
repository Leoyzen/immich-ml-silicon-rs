use std::time::Duration;
use base64::Engine;
use serde::Deserialize;

const MULTIMODAL_EMBEDDING_URL: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding";
const OCR_URL: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";
const MAX_IMAGE_BYTES: usize = 3 * 1024 * 1024; // 3MB

#[derive(Debug)]
pub enum DashScopeError {
    Api { status: u16, message: String },
    Network(String),
    Parse(String),
}

impl DashScopeError {
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Api { status, .. } => *status == 429 || *status >= 500,
            Self::Network(_) => true,
            Self::Parse(_) => false,
        }
    }
}

impl std::fmt::Display for DashScopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api { status, message } => write!(f, "DashScope API {}: {}", status, message),
            Self::Network(msg) => write!(f, "Network error: {}", msg),
            Self::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for DashScopeError {}

// Re-export OcrResult from the backends crate.
pub use immich_ml_backends::OcrResult;

impl From<DashScopeError> for immich_ml_backends::BackendError {
    fn from(e: DashScopeError) -> Self {
        match e {
            DashScopeError::Api { status, message } => immich_ml_backends::BackendError::Api { status, message },
            DashScopeError::Network(msg) => immich_ml_backends::BackendError::Network(msg),
            DashScopeError::Parse(msg) => immich_ml_backends::BackendError::Other(msg),
        }
    }
}

pub struct DashScopeClient {
    http: reqwest::Client,
    api_key: String,
    clip_model: String,
    ocr_model: String,
    clip_dim: usize,
}

impl DashScopeClient {
    pub fn new(api_key: String, clip_model: String, ocr_model: String, clip_dim: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client");
        Self { http, api_key, clip_model, ocr_model, clip_dim }
    }

    /// CLIP visual encoding: embed an image into a clip_dim-d vector.
    pub async fn clip_visual(&self, image_bytes: &[u8]) -> Result<Vec<f32>, DashScopeError> {
        let data_uri = prepare_image_data_uri(image_bytes)?;

        let body = serde_json::json!({
            "model": &self.clip_model,
            "input": {
                "contents": [{"image": data_uri}]
            },
            "parameters": {
                "enable_fusion": true,
                "dimension": self.clip_dim
            }
        });

        let resp = self.post_api(MULTIMODAL_EMBEDDING_URL, &body).await?;
        parse_embedding_response(resp)
    }

    /// CLIP textual encoding: embed text into a clip_dim-d vector.
    pub async fn clip_textual(&self, text: &str) -> Result<Vec<f32>, DashScopeError> {
        let body = serde_json::json!({
            "model": &self.clip_model,
            "input": {
                "contents": [{"text": text}]
            },
            "parameters": {
                "enable_fusion": true,
                "dimension": self.clip_dim
            }
        });

        let resp = self.post_api(MULTIMODAL_EMBEDDING_URL, &body).await?;
        parse_embedding_response(resp)
    }

    /// OCR: extract text from image via qwen-vl-ocr.
    pub async fn run_ocr(&self, image_bytes: &[u8]) -> Result<OcrResult, DashScopeError> {
        let data_uri = prepare_image_data_uri(image_bytes)?;

        let body = serde_json::json!({
            "model": &self.ocr_model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_uri},
                        {"type": "text", "text": "Read all text in the image."}
                    ]
                }]
            }
        });

        let resp = self.post_api(OCR_URL, &body).await?;
        parse_ocr_response(resp)
    }

    async fn post_api(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, DashScopeError> {
        let resp = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| DashScopeError::Network(e.to_string()))?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| DashScopeError::Network(e.to_string()))?;

        if status != 200 {
            return Err(DashScopeError::Api {
                status,
                message: text,
            });
        }

        serde_json::from_str(&text).map_err(|e| DashScopeError::Parse(format!("{}: {}", e, &text[..text.len().min(500)])))
    }
}

// --- Helpers ---

fn prepare_image_data_uri(image_bytes: &[u8]) -> Result<String, DashScopeError> {
    let bytes = if image_bytes.len() > MAX_IMAGE_BYTES {
        resize_image(image_bytes)?
    } else {
        image_bytes.to_vec()
    };

    let format = image::guess_format(&bytes).unwrap_or(image::ImageFormat::Jpeg);
    let mime = match format {
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::WebP => "image/webp",
        image::ImageFormat::Gif => "image/gif",
        _ => "image/jpeg",
    };

    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, b64))
}

fn resize_image(image_bytes: &[u8]) -> Result<Vec<u8>, DashScopeError> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| DashScopeError::Parse(format!("Image decode: {}", e)))?;

    let mut current = img;
    let mut quality = 85u8;

    loop {
        let mut buf = std::io::Cursor::new(Vec::new());
        current
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .map_err(|e| DashScopeError::Parse(format!("Image encode: {}", e)))?;

        let encoded = buf.into_inner();
        if encoded.len() <= MAX_IMAGE_BYTES || quality <= 20 {
            return Ok(encoded);
        }

        // Resize to 75% and lower quality
        let new_w = (current.width() as f32 * 0.75) as u32;
        let new_h = (current.height() as f32 * 0.75) as u32;
        current = current.resize(new_w.max(1), new_h.max(1), image::imageops::FilterType::Lanczos3);
        quality -= 10;
    }
}

// --- Response parsers ---

#[derive(Deserialize)]
struct EmbeddingResponse {
    output: EmbeddingOutput,
}

#[derive(Deserialize)]
struct EmbeddingOutput {
    embeddings: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

fn parse_embedding_response(resp: serde_json::Value) -> Result<Vec<f32>, DashScopeError> {
    let parsed: EmbeddingResponse = serde_json::from_value(resp)
        .map_err(|e| DashScopeError::Parse(format!("Embedding response: {}", e)))?;

    parsed
        .output
        .embeddings
        .into_iter()
        .next()
        .map(|item| item.embedding)
        .ok_or_else(|| DashScopeError::Parse("No embeddings in response".into()))
}

/// Parse OCR response from qwen-vl-ocr.
/// Format TBD — spike 1.6 will determine the actual response structure.
/// For now, we try to extract text from the response and synthesize bounding boxes.
fn parse_ocr_response(resp: serde_json::Value) -> Result<OcrResult, DashScopeError> {
    let mut result = OcrResult::default();

    // Try to navigate the response structure:
    // {output: {choices: [{message: {content: [{text: "..."}]}}]}}
    let output = resp.get("output")
        .ok_or_else(|| DashScopeError::Parse("Missing 'output' in OCR response".into()))?;

    // Try choices array first (multimodal-generation format)
    if let Some(choices) = output.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            if let Some(message) = choice.get("message") {
                if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
                    for item in content {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            // Split text into lines, synthesize normalized boxes
                            for (i, line) in text.lines().enumerate() {
                                let line = line.trim();
                                if line.is_empty() {
                                    continue;
                                }
                                result.text.push(line.to_string());
                                // Synthesize box: full width, line-height slice
                                let y_start = (i as f64) / (text.lines().count() as f64);
                                let y_end = ((i + 1) as f64) / (text.lines().count() as f64);
                                result.box_coords.extend_from_slice(&[
                                    0.0, y_start,
                                    1.0, y_start,
                                    1.0, y_end,
                                    0.0, y_end,
                                ]);
                                result.box_score.push(0.9);
                                result.text_score.push(0.9);
                            }
                        }
                    }
                } else if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
                    // content might be a plain string
                    for (i, line) in text.lines().enumerate() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        result.text.push(line.to_string());
                        let y_start = (i as f64) / (text.lines().count() as f64);
                        let y_end = ((i + 1) as f64) / (text.lines().count() as f64);
                        result.box_coords.extend_from_slice(&[
                            0.0, y_start,
                            1.0, y_start,
                            1.0, y_end,
                            0.0, y_end,
                        ]);
                        result.box_score.push(0.9);
                        result.text_score.push(0.9);
                    }
                }
            }
        }
    }

    Ok(result)
}

// ── Trait implementations ─────────────────────────────────────

#[async_trait::async_trait]
impl immich_ml_backends::ClipBackend for DashScopeClient {
    async fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, immich_ml_backends::BackendError> {
        DashScopeClient::clip_visual(self, image_bytes).await.map_err(Into::into)
    }
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>, immich_ml_backends::BackendError> {
        DashScopeClient::clip_textual(self, text).await.map_err(Into::into)
    }
}

#[async_trait::async_trait]
impl immich_ml_backends::OcrBackend for DashScopeClient {
    async fn recognize(&self, image_bytes: &[u8]) -> Result<OcrResult, immich_ml_backends::BackendError> {
        DashScopeClient::run_ocr(self, image_bytes).await.map_err(Into::into)
    }
    fn has_bounding_boxes(&self) -> bool { false }
}
