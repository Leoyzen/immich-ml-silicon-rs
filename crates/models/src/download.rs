//! HuggingFace model download for face recognition models (buffalo_l).

use std::path::{Path, PathBuf};

const HF_BASE: &str = "https://huggingface.co/immich-app/buffalo_l/resolve/main";

/// Expected model files for buffalo_l
pub const DETECTION_MODEL: &str = "det_10g.onnx";
pub const RECOGNITION_MODEL: &str = "w600k_mbf23.onnx";

/// Download buffalo_l model files if not already cached.
pub async fn ensure_models(cache_dir: &Path) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("Create cache dir: {}", e))?;

    let det_path = cache_dir.join(DETECTION_MODEL);
    let rec_path = cache_dir.join(RECOGNITION_MODEL);

    if !det_path.exists() {
        download_file(&format!("{}/{}", HF_BASE, DETECTION_MODEL), &det_path).await?;
    }
    if !rec_path.exists() {
        download_file(&format!("{}/{}", HF_BASE, RECOGNITION_MODEL), &rec_path).await?;
    }

    Ok((det_path, rec_path))
}

async fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    tracing::info!("Downloading {} → {:?}", url, dest);
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    let resp = client.get(url)
        .header("User-Agent", "immich-ml-rust/0.1")
        .send()
        .await
        .map_err(|e| format!("Download request: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Download failed: HTTP {}", resp.status()));
    }

    let bytes = resp.bytes().await.map_err(|e| format!("Download body: {}", e))?;
    std::fs::write(dest, &bytes).map_err(|e| format!("Write file: {}", e))?;

    tracing::info!("Downloaded {} ({} bytes)", dest.display(), bytes.len());
    Ok(())
}
