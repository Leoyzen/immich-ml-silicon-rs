# End-to-End Tests

This directory contains an end-to-end test script that exercises the full
immich-ml-rust server stack.

## Usage

```bash
# From the project root:
./tests/e2e.sh
```

## What it does

1. **Builds** the server (`cargo build -p immich-ml-server`).
2. **Downloads** two test images from Unsplash (a face photo and a bookstore
   photo with visible text) into `tests/test-face.jpg` and `tests/test-text.jpg`.
   These are cached locally — subsequent runs reuse them.
3. **Starts** the server in the background on port 3003 with the appropriate
   environment variables.
4. **Waits** for the server to become ready (polls `/ping`).
5. **Runs** five tests:
   - **Face detection + recognition** — sends a face photo, validates the
     response contains an array of detected faces with `boundingBox`,
     `embedding` (512-float JSON string), `score`, and `imageHeight`/`imageWidth`.
   - **CLIP visual** — sends an image, validates a 512-float embedding string
     under the `clip` key with image dimensions.
   - **CLIP textual** — sends text, validates a 512-float embedding string
     under the `clip` key with *no* image dimensions.
   - **OCR** — sends a text-heavy image, validates `text`, `box`, `boxScore`,
     and `textScore` arrays.
   - **Ping** — GET `/ping` returns `"pong"`.
6. **Cleans up** — kills the server process on exit.

## Requirements

- `curl` — for HTTP requests and image downloads
- `python3` — for JSON validation
- `cargo` — for building and running the server
- Models present in `model-cache/` (`det_10g.onnx`, `w600k_mbf23.onnx`)
- DashScope API key (hardcoded in the script for this test environment)

## Environment Variables

The script sets these when launching the server:

| Variable | Value |
|---|---|
| `DASHSCOPE_API_KEY` | API key for cloud CLIP/OCR backends |
| `IMMICH_ML_PORT` | `3003` |
| `IMMICH_ML_CACHE_DIR` | `./model-cache` |
| `IMMICH_ML_DEVICE` | `coreml` |
| `IMMICH_ML_DET_MODEL` | `./model-cache/det_10g.onnx` |
| `IMMICH_ML_REC_MODEL` | `./model-cache/w600k_mbf23.onnx` |

## Output

The script prints `PASS` or `FAIL` for each test with details on failure.
A summary is printed at the end showing the total pass/fail count.
The server's stdout/stderr is captured to `tests/server.log` for debugging.

Exit code is `0` if all tests pass, `1` otherwise.
