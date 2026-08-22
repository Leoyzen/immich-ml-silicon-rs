use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::Json as AxumJson;
use serde_json::json;

use crate::schema::{InferenceEntry, Payload, PipelineRequest, PipelineEntry};
use crate::state::AppState;
use immich_ml_backends::{BackendError, ImageInput, DecodedImage, FaceDetectionOutput};

pub async fn ping() -> &'static str {
    "pong"
}

pub async fn predict(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let request_id = format!("{:08x}", state.request_counter.fetch_add(1, Ordering::Relaxed));

    // 1. Parse multipart form data
    let mut entries_json: Option<String> = None;
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "entries" => {
                entries_json = Some(field.text().await.unwrap_or_default());
            }
            "image" => {
                image_bytes = Some(field.bytes().await.map(|b| b.to_vec()).unwrap_or_default());
            }
            "text" => {
                text = Some(field.text().await.unwrap_or_default());
            }
            _ => { /* skip unknown fields */ }
        }
    }

    let entries_str = match entries_json {
        Some(s) => s,
        None => return (StatusCode::BAD_REQUEST, "Missing 'entries' field").into_response(),
    };

    // 2. Parse entries JSON
    let pipeline: PipelineRequest = match serde_json::from_str(&entries_str) {
        Ok(p) => p,
        Err(e) => return (StatusCode::UNPROCESSABLE_ENTITY, format!("Invalid entries JSON: {}", e)).into_response(),
    };

    // 3. Determine payload and image dimensions
    let is_image = image_bytes.is_some();
    let (payload, image_dims, decoded_image) = if let Some(bytes) = image_bytes {
        // Decode image once to get both dimensions AND RGBA pixels (with HEIC support on macOS).
        match immich_ml_imaging::decode_image(&bytes) {
            Ok((rgba, w, h)) => {
                if w == 0 || h == 0 {
                    return (StatusCode::BAD_REQUEST, "Image has zero width or height").into_response();
                }
                let decoded = DecodedImage { rgba, width: w, height: h };
                (Payload::Image(bytes), Some((w, h)), Some(decoded))
            }
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("Image decode error: {}", e)).into_response();
            }
        }
    } else if let Some(t) = text {
        (Payload::Text(t), None, None)
    } else {
        return (StatusCode::BAD_REQUEST, "Either image or text must be provided").into_response();
    };

    // 4. Flatten entries and split by dependency
    let (without_deps, with_deps) = flatten_entries(pipeline);
    let tasks: Vec<String> = without_deps.iter().chain(with_deps.iter())
        .map(|e| format!("{}/{}", e.task, e.r#type)).collect();
    let tasks_str = tasks.join(", ");
    tracing::info!("[{}] /predict request: tasks=[{}], payload={}", request_id, tasks_str,
        if is_image { "image" } else { "text" });

    // 5. Build response
    let mut response: HashMap<String, serde_json::Value> = HashMap::new();

    // Track intermediate outputs (e.g., face detection results for recognition)
    let mut face_detection: Option<FaceDetectionOutput> = None;

    // Run without_deps
    for entry in &without_deps {
        match (entry.task.as_str(), entry.type_as_str()) {
            ("clip", "visual") => {
                match run_clip_visual(&state, &payload).await {
                    Ok(embedding) => {
                        let embedding_str = serde_json::to_string(&embedding).unwrap();
                        response.insert("clip".to_string(), json!(embedding_str));
                    }
                    Err(status_msg) => return status_msg.into_response(),
                }
            }
            ("clip", "textual") => {
                match run_clip_textual(&state, &payload).await {
                    Ok(embedding) => {
                        let embedding_str = serde_json::to_string(&embedding).unwrap();
                        response.insert("clip".to_string(), json!(embedding_str));
                    }
                    Err(status_msg) => return status_msg.into_response(),
                }
            }
            ("ocr", _) => {
                match run_ocr(&state, &payload).await {
                    Ok(ocr_result) => {
                        response.insert("ocr".to_string(), json!(ocr_result));
                    }
                    Err(status_msg) => return status_msg.into_response(),
                }
            }
            ("facial-recognition", "detection") => {
                let image_input = match &payload {
                    Payload::Image(bytes) => ImageInput {
                        bytes: bytes.clone(),
                        width: image_dims.map(|(w, _)| w).unwrap_or(0),
                        height: image_dims.map(|(_, h)| h).unwrap_or(0),
                        decoded: decoded_image.clone(),
                    },
                    Payload::Text(_) => {
                        return (StatusCode::BAD_REQUEST, "Face detection requires image").into_response();
                    }
                };
                let min_score = entry.options.get("minScore")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(0.5);
                match run_face_detection(&state, &image_input, min_score).await {
                    Ok(result) => {
                        face_detection = Some(result);
                    }
                    Err(status_msg) => return status_msg.into_response(),
                }
            }
            _ => {
                tracing::warn!("[{}] Unknown task/type: {}/{}", request_id, entry.task, entry.r#type);
            }
        }
    }

    // Run with_deps
    for entry in &with_deps {
        match (entry.task.as_str(), entry.type_as_str()) {
            ("facial-recognition", "recognition") => {
                let detection = match &face_detection {
                    Some(d) => d,
                    None => {
                        tracing::warn!("[{}] Face recognition requested but no detection output available", request_id);
                        response.insert("facial-recognition".to_string(), json!([]));
                        continue;
                    }
                };
                let image_input = match &payload {
                    Payload::Image(bytes) => ImageInput {
                        bytes: bytes.clone(),
                        width: image_dims.map(|(w, _)| w).unwrap_or(0),
                        height: image_dims.map(|(_, h)| h).unwrap_or(0),
                        decoded: decoded_image.clone(),
                    },
                    Payload::Text(_) => {
                        return (StatusCode::BAD_REQUEST, "Face recognition requires image").into_response();
                    }
                };
                match run_face_recognition(&state, &image_input, detection).await {
                    Ok(faces) => {
                        response.insert("facial-recognition".to_string(), json!(faces));
                    }
                    Err(status_msg) => return status_msg.into_response(),
                }
            }
            // OCR recognition is handled by ocr/detection in the without_deps loop — skip silently
            ("ocr", "recognition") => {}
            _ => {
                tracing::warn!("[{}] Unknown dependent task: {}/{}", request_id, entry.task, entry.r#type);
            }
        }
    }

    // 6. Add imageHeight/imageWidth (top-level, only for image payloads)
    if let Some((w, h)) = image_dims {
        response.insert("imageHeight".to_string(), json!(h));
        response.insert("imageWidth".to_string(), json!(w));
    }

    tracing::info!("[{}] /predict done: tasks=[{}], keys=[{}]", request_id, tasks_str, response.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(", "));
    Json(response).into_response()
}

// --- Helpers ---

impl InferenceEntry {
    fn type_as_str(&self) -> &str {
        &self.r#type
    }
}

fn flatten_entries(pipeline: PipelineRequest) -> (Vec<InferenceEntry>, Vec<InferenceEntry>) {
    let mut without_deps = Vec::new();
    let mut with_deps = Vec::new();

    for (task, types) in pipeline {
        for (r#type, entry) in types {
            let inference_entry = InferenceEntry {
                task: task.clone(),
                r#type: r#type.clone(),
                model_name: entry.model_name,
                options: entry.options.unwrap_or_default(),
            };

            // Recognition depends on detection; everything else is independent
            let has_deps = r#type == "recognition";
            if has_deps {
                with_deps.push(inference_entry);
            } else {
                without_deps.push(inference_entry);
            }
        }
    }

    (without_deps, with_deps)
}

type ErrorResponse = (StatusCode, String);

async fn run_clip_visual(state: &AppState, payload: &Payload) -> Result<Vec<f32>, ErrorResponse> {
    if let Some(ref batcher) = state.clip_batcher {
        // Use batcher for DashScope — bypasses call_with_retry since batch
        // handles its own errors internally. Circuit breaker check is still done here.
        let image_bytes = match payload {
            Payload::Image(bytes) => bytes.clone(),
            Payload::Text(_) => return Err((StatusCode::BAD_REQUEST, "CLIP visual requires image".into())),
        };

        if state.concurrency.is_tripped() {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Circuit breaker tripped".into()));
        }

        match batcher.submit(image_bytes).await {
            Ok(emb) => {
                state.concurrency.record_success();
                Ok(emb)
            }
            Err(e) => {
                state.concurrency.record_failure();
                Err((StatusCode::INTERNAL_SERVER_ERROR, format!("CLIP batch failed: {}", e)))
            }
        }
    } else {
        // Non-batching backends: use existing call_with_retry path
        let image_bytes = match payload {
            Payload::Image(bytes) => bytes.as_slice(),
            Payload::Text(_) => return Err((StatusCode::BAD_REQUEST, "CLIP visual requires image".into())),
        };
        call_with_retry(state, || async {
            state.clip.encode_image(image_bytes).await
        }).await
    }
}

async fn run_clip_textual(state: &AppState, payload: &Payload) -> Result<Vec<f32>, ErrorResponse> {
    let text = match payload {
        Payload::Text(t) => t.as_str(),
        Payload::Image(_) => return Err((StatusCode::BAD_REQUEST, "CLIP textual requires text".into())),
    };
    call_with_retry(state, || async {
        state.clip.encode_text(text).await
    }).await
}

async fn run_ocr(state: &AppState, payload: &Payload) -> Result<immich_ml_backends::OcrResult, ErrorResponse> {
    let image_bytes = match payload {
        Payload::Image(bytes) => bytes.as_slice(),
        Payload::Text(_) => return Err((StatusCode::BAD_REQUEST, "OCR requires image".into())),
    };
    // OCR degrades gracefully — return empty result on failure, not 500
    match call_with_retry(state, || async {
        state.ocr.recognize(image_bytes).await
    }).await {
        Ok(result) => Ok(result),
        Err(_) => Ok(immich_ml_backends::OcrResult::default()),
    }
}

async fn run_face_detection(
    state: &AppState,
    image_input: &ImageInput,
    min_score: f32,
) -> Result<FaceDetectionOutput, ErrorResponse> {
    call_with_retry(state, || async {
        state.face_detector.detect(image_input, min_score).await
    }).await
}

async fn run_face_recognition(
    state: &AppState,
    image_input: &ImageInput,
    detection: &FaceDetectionOutput,
) -> Result<Vec<immich_ml_backends::DetectedFace>, ErrorResponse> {
    call_with_retry(state, || async {
        state.face_recognizer.recognize(image_input, detection).await
    }).await
}

/// Call a backend function with concurrency control, retry, timeout, and circuit breaker.
async fn call_with_retry<F, Fut, T>(
    state: &AppState,
    f: F,
) -> Result<T, ErrorResponse>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, BackendError>>,
{
    if state.concurrency.is_tripped() {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "Circuit breaker tripped".into()));
    }

    let mut last_err = None;
    let timeout = state.concurrency.timeout();

    for attempt in 0..=state.concurrency.max_retries() {
        // Acquire permit per-attempt so it is not held during backoff sleep
        let _permit = state.concurrency.acquire().await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

        match tokio::time::timeout(timeout, f()).await {
            Ok(Ok(result)) => {
                state.concurrency.record_success();
                return Ok(result);
            }
            Ok(Err(e)) => {
                if !e.is_retriable() || attempt == state.concurrency.max_retries() {
                    last_err = Some(e);
                    break;
                }
                last_err = Some(e);
            }
            Err(_elapsed) => {
                // Timeout is retriable
                let e = BackendError::Network("Request timed out".into());
                if !e.is_retriable() || attempt == state.concurrency.max_retries() {
                    last_err = Some(e);
                    break;
                }
                last_err = Some(e);
            }
        }

        // Release permit before sleeping so other tasks can proceed during backoff
        drop(_permit);

        let delay = state.concurrency.backoff_delay(attempt);
        tracing::warn!(
            "Backend call failed (attempt {}), retrying in {:?}: {}",
            attempt, delay, last_err.as_ref().unwrap()
        );
        tokio::time::sleep(delay).await;
    }

    state.concurrency.record_failure();
    Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("Backend failed: {}", last_err.map(|e| e.to_string()).unwrap_or_default()),
    ))
}
