//! Apple Vision backends (macOS only).
//!
//! Uses the macOS Vision framework via `objc2-vision` to perform on-device
//! text recognition and face detection. On non-macOS targets stubs are
//! provided that return an error at runtime.

use immich_ml_backends::{BackendError, FaceDetectionBackend, FaceDetectionOutput, ImageInput, OcrBackend, OcrResult};

// ── VisionOcrBackend ───────────────────────────────────────────

pub struct VisionOcrBackend;

impl VisionOcrBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VisionOcrBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ── VisionFaceDetector ─────────────────────────────────────────

pub struct VisionFaceDetector;

impl VisionFaceDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VisionFaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── macOS implementation ───────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use immich_ml_backends::{BackendError, FaceDetectionOutput, OcrResult};
    use objc2::rc::autoreleasepool;
    use objc2::AnyThread; // for VNImageRequestHandler::alloc()
    use objc2::runtime::AnyObject;
    use objc2_foundation::{NSArray, NSData, NSDictionary, NSString};
    use objc2_vision::{
        VNDetectFaceRectanglesRequest, VNDetectedObjectObservation,
        VNImageRequestHandler, VNObservation, VNRecognizeTextRequest,
        VNRequestTextRecognitionLevel,
    };

    /// Minimum confidence threshold for accepting a recognition candidate.
    const MIN_OCR_CONFIDENCE: f32 = 0.01;

    // ── OCR ────────────────────────────────────────────────────

    pub fn vision_ocr_sync(image_bytes: &[u8]) -> Result<OcrResult, BackendError> {
        autoreleasepool(|_pool| {
            // Build the text-recognition request.
            let request = VNRecognizeTextRequest::new();
            request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);

            // Set recognition languages (needs NSArray<NSString>).
            let langs = [
                NSString::from_str("zh-Hans"),
                NSString::from_str("zh-Hant"),
                NSString::from_str("en-US"),
            ];
            let lang_arr = NSArray::from_retained_slice(&langs);
            request.setRecognitionLanguages(&lang_arr);

            request.setUsesLanguageCorrection(true);
            // No setMinimumTextConfidence API — filter manually below.

            // Wrap image bytes in NSData and create a handler.
            let ns_data = NSData::with_bytes(image_bytes);
            let options = NSDictionary::<NSString, AnyObject>::new();
            let handler = VNImageRequestHandler::initWithData_options(
                VNImageRequestHandler::alloc(),
                &ns_data,
                &options,
            );

            // Perform the request synchronously.
            // Upcast VNRecognizeTextRequest → VNRequest for the NSArray.
            let vn_request = request.clone().into_super().into_super();
            let requests = NSArray::from_retained_slice(&[vn_request]);
            handler
                .performRequests_error(&requests)
                .map_err(|err| BackendError::Other(format!("Vision OCR failed: {:?}", err)))?;

            // Collect results.
            let mut result = OcrResult::default();
            if let Some(results_arr) = request.results() {
                for obs in results_arr.iter() {
                    let candidates = obs.topCandidates(1);
                    if let Some(candidate) = candidates.firstObject() {
                        let confidence = candidate.confidence();
                        if confidence < MIN_OCR_CONFIDENCE {
                            continue;
                        }

                        let text = candidate.string().to_string();

                        // Vision returns CGRect with lower-left origin, normalized [0,1].
                        // Access boundingBox via AsRef<VNDetectedObjectObservation>.
                        let bbox = unsafe {
                            let det: &VNDetectedObjectObservation = obs.as_ref();
                            det.boundingBox()
                        };
                        let x1 = bbox.origin.x as f64;
                        let x2 = (bbox.origin.x + bbox.size.width) as f64;
                        // Flip Y axis: top-left origin.
                        let y1 = 1.0 - ((bbox.origin.y + bbox.size.height) as f64); // top
                        let y2 = 1.0 - (bbox.origin.y as f64); // bottom

                        // Flatten as TL→TR→BR→BL: [x1,y1, x2,y1, x2,y2, x1,y2]
                        result.text.push(text);
                        result
                            .box_coords
                            .extend_from_slice(&[x1, y1, x2, y1, x2, y2, x1, y2]);
                        result.box_score.push(confidence as f64);
                        result.text_score.push(confidence as f64);
                    }
                }
            }

            Ok(result)
        })
    }

    // ── Face Detection ─────────────────────────────────────────

    pub fn vision_face_detect_sync(
        image_bytes: &[u8],
        width: u32,
        height: u32,
        min_score: f32,
    ) -> Result<FaceDetectionOutput, BackendError> {
        autoreleasepool(|_pool| {
            // Build the face-detection request.
            let request = unsafe { VNDetectFaceRectanglesRequest::new() };

            // Wrap image bytes in NSData and create a handler.
            let ns_data = NSData::with_bytes(image_bytes);
            let options = NSDictionary::<NSString, AnyObject>::new();
            let handler = VNImageRequestHandler::initWithData_options(
                VNImageRequestHandler::alloc(),
                &ns_data,
                &options,
            );

            // Perform the request synchronously.
            // Upcast VNDetectFaceRectanglesRequest → VNRequest for the NSArray.
            let vn_request = request.clone().into_super().into_super();
            let requests = NSArray::from_retained_slice(&[vn_request]);
            handler
                .performRequests_error(&requests)
                .map_err(|err| BackendError::Other(format!("Vision face detection failed: {:?}", err)))?;

            // Collect results.
            let mut output = FaceDetectionOutput::default();
            if let Some(results_arr) = unsafe { request.results() } {
                for obs in results_arr.iter() {
                    // confidence() is on VNObservation (superclass).
                    let confidence = unsafe {
                        let vn_obs: &VNObservation = obs.as_ref();
                        vn_obs.confidence()
                    };
                    if confidence < min_score {
                        continue;
                    }

                    // boundingBox() is on VNDetectedObjectObservation (superclass).
                    // Returns normalized [0,1] with lower-left origin.
                    let bbox = unsafe {
                        let det: &VNDetectedObjectObservation = obs.as_ref();
                        det.boundingBox()
                    };

                    // Convert to pixel coordinates with top-left origin.
                    let w = width as f64;
                    let h = height as f64;
                    let x1 = (bbox.origin.x as f64) * w;
                    let x2 = ((bbox.origin.x + bbox.size.width) as f64) * w;
                    // Flip Y: top-left origin.
                    let y1 = (1.0 - ((bbox.origin.y + bbox.size.height) as f64)) * h; // top
                    let y2 = (1.0 - (bbox.origin.y as f64)) * h; // bottom

                    output.boxes.push([x1 as f32, y1 as f32, x2 as f32, y2 as f32]);
                    output.scores.push(confidence);
                    // VNDetectFaceRectanglesRequest does not provide landmarks.
                }
            }

            Ok(output)
        })
    }
}

// ── OcrBackend impl ────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[async_trait::async_trait]
impl OcrBackend for VisionOcrBackend {
    async fn recognize(&self, image_bytes: &[u8]) -> Result<OcrResult, BackendError> {
        let bytes = image_bytes.to_vec();
        tokio::task::spawn_blocking(move || macos::vision_ocr_sync(&bytes))
            .await
            .map_err(|e| BackendError::Other(format!("Join error: {}", e)))?
    }

    fn has_bounding_boxes(&self) -> bool {
        true
    }
}

#[cfg(not(target_os = "macos"))]
#[async_trait::async_trait]
impl OcrBackend for VisionOcrBackend {
    async fn recognize(&self, _image_bytes: &[u8]) -> Result<OcrResult, BackendError> {
        Err(BackendError::Other(
            "Vision backend requires macOS".into(),
        ))
    }

    fn has_bounding_boxes(&self) -> bool {
        true
    }
}

// ── FaceDetectionBackend impl ──────────────────────────────────

#[cfg(target_os = "macos")]
#[async_trait::async_trait]
impl FaceDetectionBackend for VisionFaceDetector {
    async fn detect(
        &self,
        image: &ImageInput,
        min_score: f32,
    ) -> Result<FaceDetectionOutput, BackendError> {
        let bytes = image.bytes.clone();
        let width = image.width;
        let height = image.height;
        tokio::task::spawn_blocking(move || {
            macos::vision_face_detect_sync(&bytes, width, height, min_score)
        })
        .await
        .map_err(|e| BackendError::Other(format!("Join error: {}", e)))?
    }
}

#[cfg(not(target_os = "macos"))]
#[async_trait::async_trait]
impl FaceDetectionBackend for VisionFaceDetector {
    async fn detect(
        &self,
        _image: &ImageInput,
        _min_score: f32,
    ) -> Result<FaceDetectionOutput, BackendError> {
        Err(BackendError::Other(
            "Vision backend requires macOS".into(),
        ))
    }
}

// Safety: Both backends hold no state; Vision objects are created fresh
// per-call inside spawn_blocking.
unsafe impl Send for VisionOcrBackend {}
unsafe impl Sync for VisionOcrBackend {}
unsafe impl Send for VisionFaceDetector {}
unsafe impl Sync for VisionFaceDetector {}
