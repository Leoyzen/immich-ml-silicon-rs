# immich-ml-silicon-rs

A Rust-native ML service for [immich](https://github.com/immich-app/immich) on Apple Silicon. Drop-in replacement for `immich-machine-learning` with hardware-accelerated inference and cloud hybrid architecture.

## Architecture

| Task | Backend | Engine | Latency |
|------|---------|--------|---------|
| Face detection | ONNX (CoreML EP) or Apple Vision | SCRFD / VNDetectFaceRectangles | ~7ms |
| Face recognition | ONNX (CoreML EP) | ArcFace 512-d | ~5ms/face |
| CLIP visual+textual | Alibaba DashScope | qwen3-vl-embedding 512-d | ~500ms (batched) |
| OCR | Apple Vision or DashScope | VNRecognizeTextRequest / qwen-vl-ocr | ~300ms |

**Hybrid design:** Face detection/recognition runs locally on CoreML GPU (7.54ms/inference). CLIP and OCR offload to Alibaba Cloud DashScope API, with a batch coalescer that merges up to 10 images per API call for 3-5x throughput.

**Apple hardware acceleration:**
- ImageIO for JPEG/PNG/HEIC/HEIF decoding (hardware-accelerated, EXIF auto-rotation)
- CoreML EP for ONNX face inference (GPU)
- Vision framework for OCR and optional face detection (ANE)
- Static ONNX models (fixed dimensions) for full GPU utilization

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1+)
- [Rust](https://rustup.rs/) (stable, 1.75+)
- [direnv](https://direnv.net/) (optional, for env management)
- DashScope API key ([get one](https://dashscope.console.aliyun.com/))
- immich face models (buffalo_s or buffalo_l)

### Build

```bash
git clone https://github.com/Leoyzen/immich-ml-silicon-rs.git
cd immich-ml-silicon-rs
cargo build --release
```

### Configure

```bash
cp .envrc.example .envrc
# Edit .envrc with your DashScope API key and model paths
direnv allow
```

### Download face models

```bash
mkdir -p model-cache
# From your existing immich installation:
rsync -avz nas:/mnt/cache/appdata/immich/machine-learning/models/facial-recognition/buffalo_s/ model-cache/
# Or from HuggingFace: immich-app/buffalo_s
```

**Important:** ONNX models must have static (fixed) dimensions for CoreML GPU compilation. Use the included Python script to fix dynamic dimensions:

```bash
python3 -c "
import onnx
m = onnx.load('model-cache/det_10g.onnx')
for inp in m.graph.input:
    for dim in inp.type.tensor_type.shape.dim:
        if dim.dim_value == 0 and dim.dim_param:
            dim.ClearField('dim_param')
            dim.dim_value = 640 if 'height' in str(dim) or 'width' in str(dim) else 1
onnx.save(m, 'model-cache/det_10g_fixed.onnx')
"
```

### Run

```bash
direnv exec . ./target/release/immich-ml-server
# Server listens on 0.0.0.0:3003
```

### Configure immich

1. In immich Admin Settings → Machine Learning:
   - Set URL to `http://<this-machine-ip>:3003`
   - Set CLIP model to `ViT-B-32__openai` (passthrough — actual model is qwen3-vl-embedding)
2. Trigger ML jobs (Detect Faces, Encode Clips, Read Text)

## Configuration

All config via environment variables (see `.envrc.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | required | Alibaba DashScope API key |
| `IMMICH_ML_PORT` | 3003 | HTTP listen port |
| `IMMICH_ML_DEVICE` | coreml | ONNX execution provider |
| `IMMICH_ML_DET_MODEL_PATH` | required | Face detection ONNX model path |
| `IMMICH_ML_REC_MODEL_PATH` | required | Face recognition ONNX model path |
| `IMMICH_ML_FACE_DETECTION_BACKEND` | onnx | `onnx` or `vision` |
| `IMMICH_ML_FACE_RECOGNITION_BACKEND` | onnx | `onnx` (only option) |
| `IMMICH_ML_CLIP_BACKEND` | dashscope | `dashscope` (only option) |
| `IMMICH_ML_CLIP_DIM` | 512 | CLIP embedding dimension (match pgvector) |
| `IMMICH_ML_OCR_BACKEND` | vision | `vision` or `dashscope` |
| `IMMICH_ML_ONNX_SESSIONS` | 2 | Parallel ONNX session pool size (1-4) |
| `IMMICH_ML_MAX_CONCURRENCY` | 10 | Max concurrent DashScope API requests |
| `IMMICH_ML_CLIP_BATCH_SIZE` | 10 | Images per batched DashScope call (1-10) |
| `IMMICH_ML_CLIP_BATCH_INTERVAL_MS` | 50 | Batch flush timeout |
| `RUST_LOG` | info | Log level |

## Deployment

### launchd (macOS auto-start)

```bash
cp deploy/com.leoyzen.immich-ml-silicon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.leoyzen.immich-ml-silicon.plist
```

See `deploy/` for the plist template.

## Crate Structure

```
crates/
  backends/  — Trait abstractions (FaceDetection/Recognition/Clip/Ocr)
  models/    — ONNX face detection+recognition (CoreML EP, multi-session pool)
  cloud/     — DashScope CLIP + OCR client (batch support)
  vision/    — Apple Vision OCR + face detection (objc2-vision)
  imaging/   — ImageIO hardware-accelerated image decoding (HEIC support)
  server/    — axum HTTP /predict + /ping, CLIP batch coalescer
spikes/
  ort-coreml/     — CoreML EP verification spike
  dashscope-api/  — DashScope API validation spike
tests/
  e2e.sh  — End-to-end integration tests
```

## Tested With

- macOS 15 (Sequoia) on Apple Silicon (M1 Mac Mini 16GB)
- immich v1.133 (single Docker container on NAS)
- Models: buffalo_s (det_10g + w600k_mbf23), fixed dimensions
- Throughput: 2300+ requests, 0 errors

## License

MIT
