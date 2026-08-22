//! Spike 1.5-1.6: Verify DashScope API for CLIP cross-modal alignment and OCR format.
//!
//! Usage:
//!   export DASHSCOPE_API_KEY=sk-...
//!   cargo run -p spike-dashscope-api -- /path/to/test_images/
//!
//! Tests:
//! 1.5 — Cross-modal alignment: image+text cosine similarity via qwen3-vl-embedding
//! 1.6 — OCR response format: call qwen-vl-ocr and document the response structure

use std::time::Instant;

const EMBEDDING_URL: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding";
const OCR_URL: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let api_key = std::env::var("DASHSCOPE_API_KEY")
        .map_err(|_| "DASHSCOPE_API_KEY not set")?;

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // --- Spike 1.5: Cross-modal alignment ---
    println!("=== Spike 1.5: Cross-modal alignment (qwen3-vl-embedding) ===\n");

    // Test text pairs (we'll use text-only since we need test images for image pairs)
    // For a proper test, provide image files as arguments
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        // Text-only test: embed multiple texts and compute pairwise similarity
        println!("No image files provided — running text-only similarity test");
        println!("For full cross-modal test, provide image paths as arguments\n");

        let test_texts = vec![
            "a beautiful sunset over the ocean",
            "golden hour sky with orange clouds",
            "a dog playing on the beach",
            "birthday cake with candles",
            "a red sports car on a highway",
        ];

        let mut embeddings = Vec::new();
        for text in &test_texts {
            let emb = embed_text(&http, &api_key, text).await?;
            println!("  Text: {:40} → {} dims", &text[..text.len().min(40)], emb.len());
            embeddings.push(emb);
        }

        println!("\n  Pairwise cosine similarities (512-d):");
        for i in 0..embeddings.len() {
            for j in (i+1)..embeddings.len() {
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                println!("    [{:.3}] '{}' vs '{}'", sim,
                    &test_texts[i][..test_texts[i].len().min(30)],
                    &test_texts[j][..test_texts[j].len().min(30)]);
            }
        }
    } else {
        // Cross-modal test: embed images and matched/mismatched texts
        println!("Testing with {} image(s)\n", args.len());

        let text_descriptions = vec![
            "a sunset", "a dog", "a cake", "a car", "a person",
            "a building", "flowers", "mountains", "food", "water",
        ];

        for img_path in &args {
            println!("--- Image: {} ---", img_path);
            let img_bytes = std::fs::read(img_path)?;
            
            // Embed image at 512-d
            let img_emb_512 = embed_image(&http, &api_key, &img_bytes, 512).await?;
            println!("  Image embedding (512-d): {} values", img_emb_512.len());

            // Embed image at native dimensions (e.g., 1024)
            let img_emb_native = embed_image(&http, &api_key, &img_bytes, 1024).await?;
            println!("  Image embedding (1024-d): {} values", img_emb_native.len());

            // Compute similarity with each text description
            println!("\n  Cross-modal cosine similarities (512-d):");
            for desc in &text_descriptions {
                let text_emb = embed_text_dim(&http, &api_key, desc, 512).await?;
                let sim = cosine_similarity(&img_emb_512, &text_emb);
                let marker = if sim > 0.3 { " ← MATCH" } else { "" };
                println!("    [{:.4}] vs '{}'{}", sim, desc, marker);
            }

            // 512-d vs native dimension quality comparison
            println!("\n  Dimension truncation comparison:");
            for desc in &text_descriptions {
                let text_emb_512 = embed_text_dim(&http, &api_key, desc, 512).await?;
                let text_emb_native = embed_text_dim(&http, &api_key, desc, 1024).await?;
                let sim_512 = cosine_similarity(&img_emb_512, &text_emb_512);
                let sim_native = cosine_similarity(&img_emb_native, &text_emb_native);
                println!("    512d={:.4}  1024d={:.4}  Δ={:+.4}  '{}'",
                    sim_512, sim_native, sim_512 - sim_native, desc);
            }
            println!();
        }

        // Go/no-go assessment
        println!("=== Go/No-Go Assessment ===");
        println!("If matched pairs show cosine > 0.3: GO (use qwen3-vl-embedding)");
        println!("If matched pairs show cosine < 0.3: NO-GO (pivot to Plan B: local CPU CLIP)");
    }

    // --- Spike 1.6: OCR response format ---
    println!("\n=== Spike 1.6: OCR response format (qwen-vl-ocr) ===\n");

    if !args.is_empty() {
        // Use first image for OCR test
        let img_path = &args[0];
        println!("Testing OCR with: {}", img_path);
        let img_bytes = std::fs::read(img_path)?;
        
        let data_uri = make_data_uri(&img_bytes)?;
        let body = serde_json::json!({
            "model": "qwen-vl-ocr",
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

        let start = Instant::now();
        let resp = http.post(OCR_URL)
            .bearer_auth(&api_key)
            .json(&body)
            .send()
            .await?;
        
        let status = resp.status();
        let text = resp.text().await?;
        let elapsed = start.elapsed();

        println!("  Status: {}", status);
        println!("  Latency: {:.0}ms", elapsed.as_millis());
        println!("  Response:\n{}", pretty_print_json(&text)?);
        println!("\n  NOTE: Document the response structure above for task 7.3");
        println!("  Does it return bounding boxes? If not, synthesize normalized boxes.");
    } else {
        println!("  (Skip — no test image provided)");
    }

    println!("\n=== Spike complete ===");
    Ok(())
}

async fn embed_text(
    http: &reqwest::Client,
    api_key: &str,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    embed_text_dim(http, api_key, text, 512).await
}

async fn embed_text_dim(
    http: &reqwest::Client,
    api_key: &str,
    text: &str,
    dim: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let body = serde_json::json!({
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": text}]},
        "parameters": {"enable_fusion": true, "dimension": dim}
    });

    let resp = http.post(EMBEDDING_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    let v: serde_json::Value = resp.json().await?;
    let emb = v["output"]["embeddings"][0]["embedding"]
        .as_array()
        .ok_or("no embedding in response")?
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect();
    Ok(emb)
}

async fn embed_image(
    http: &reqwest::Client,
    api_key: &str,
    img_bytes: &[u8],
    dim: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let data_uri = make_data_uri(img_bytes)?;
    let body = serde_json::json!({
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"image": data_uri}]},
        "parameters": {"enable_fusion": true, "dimension": dim}
    });

    let resp = http.post(EMBEDDING_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    let v: serde_json::Value = resp.json().await?;
    let emb = v["output"]["embeddings"][0]["embedding"]
        .as_array()
        .ok_or("no embedding in response")?
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect();
    Ok(emb)
}

fn make_data_uri(bytes: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:image/jpeg;base64,{}", b64))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn pretty_print_json(s: &str) -> Result<String, Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(s)?;
    Ok(serde_json::to_string_pretty(&v)?)
}
