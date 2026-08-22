//! Spike 1.1-1.4: Verify ort crate CoreML EP support for face models.
//!
//! Usage:
//!   # Download model from https://huggingface.co/immich-app/buffalo_l
//!   cargo run -p spike-ort-coreml -- /path/to/det_10g.onnx [test_image.jpg]
//!
//! Tests:
//! 1.1 — Load SCRFD model with CoreMLExecutionProvider
//! 1.2 — Verify CoreML EP options (MLProgram, ComputeUnits::All, FastPrediction)
//! 1.3 — Benchmark inference latency + memory stability
//! 1.4 — Check ArcFace batch axis handling (dynamic shape)

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: spike-ort-coreml <model.onnx> [test_image.jpg]");
    let image_path = args.get(2);

    println!("=== Spike 1.1: ort crate CoreML EP ===");
    println!("Model: {}", model_path);
    println!("ort version: 2.0.0-rc.13");

    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::ep::CoreML;
    use ort::ep::coreml::{ModelFormat, SpecializationStrategy, ComputeUnits};

    // 1.1 + 1.2: Load model with CoreML EP + all options matching immich-ml's Python config:
    //   MLProgram=True, EPFlags=USE_ALL_ONLY, FastPrediction=True
    println!("\n--- Loading model with CoreML EP (MLProgram + All + FastPrediction) ---");

    let ep = CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(ComputeUnits::All)
        .with_specialization_strategy(SpecializationStrategy::FastPrediction)
        .build();

    let session_result = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([ep])?
        .commit_from_file(model_path);

    let mut session = match session_result {
        Ok(session) => {
            println!("✓ Model loaded with CoreML EP (MLProgram + All + FastPrediction) successfully");
            session
        }
        Err(e) => {
            println!("✗ Failed to load with full CoreML EP options: {}", e);
            println!("\nFallback: trying default CoreML EP...");
            
            match Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CoreML::default().build()])?
                .commit_from_file(model_path)
            {
                Ok(s) => {
                    println!("✓ Model loaded with default CoreML EP");
                    s
                }
                Err(e2) => {
                    println!("✗ CoreML EP failed entirely: {}", e2);
                    println!("\nFinal fallback: CPU EP...");
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(model_path)?
                }
            }
        }
    };

    // Print input/output info
    println!("\n--- Model I/O ---");
    for (i, input) in session.inputs().iter().enumerate() {
        let shape = input.dtype().tensor_shape();
        println!("  Input[{}]: {} dtype={}", i, input.name(), input.dtype());
        if let Some(s) = shape {
            println!("    shape: {:?}", s);
        }
    }
    for (i, output) in session.outputs().iter().enumerate() {
        println!("  Output[{}]: {} dtype={}", i, output.name(), output.dtype());
    }

    // 1.4: Check if input has dynamic batch dimension
    println!("\n--- Spike 1.4: Batch axis check ---");
    if let Some(first_input) = session.inputs().first() {
        if let Some(shape) = first_input.dtype().tensor_shape() {
            if let Some(&batch_dim) = shape.first() {
                if batch_dim == -1 || batch_dim == 0 {
                    println!("✓ Input has dynamic batch dimension (value={})", batch_dim);
                    println!("  ArcFace batch axis handled at runtime — no protobuf modification needed");
                } else if batch_dim == 1 {
                    println!("⚠ Input has fixed batch=1");
                    println!("  ArcFace recognition will need batch axis added (see task 3.3)");
                } else {
                    println!("? Input batch dimension: {}", batch_dim);
                }
            }
        }
    }

    // 1.3: Benchmark if test image provided
    if let Some(img_path) = image_path {
        println!("\n--- Spike 1.3: Benchmark ---");
        benchmark_inference(&mut session, img_path)?;
    } else {
        println!("\n(Skip benchmark — no test image provided)");
        println!("  Run with: spike-ort-coreml <model.onnx> <test.jpg>");
    }

    println!("\n=== Spike complete ===");
    Ok(())
}

fn benchmark_inference(
    session: &mut ort::session::Session,
    image_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::Array4;
    use ort::value::Tensor;

    // Load and preprocess image
    let img = image::open(image_path)?;
    let img = img.to_rgb8();
    println!("  Image: {}x{}", img.width(), img.height());

    // Letterbox to 640x640
    let size = 640u32;
    let (new_w, new_h) = if img.height() > img.width() {
        (size * img.width() / img.height(), size)
    } else {
        (size, size * img.height() / img.width())
    };
    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
    
    let mut canvas = image::RgbImage::new(size, size);
    image::imageops::overlay(&mut canvas, &resized, 0, 0);

    // Convert to NCHW float32, normalize: mean=127.5, std=128
    let mut input_data = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for y in 0..size as usize {
        for x in 0..size as usize {
            let pixel = canvas.get_pixel(x as u32, y as u32);
            input_data[[0, 0, y, x]] = (pixel[0] as f32 - 127.5) / 128.0;
            input_data[[0, 1, y, x]] = (pixel[1] as f32 - 127.5) / 128.0;
            input_data[[0, 2, y, x]] = (pixel[2] as f32 - 127.5) / 128.0;
        }
    }

    let input_tensor = Tensor::from_array(input_data)?;
    
    // Warm up
    {
        let warmup = session.run(ort::inputs![input_tensor.view()])?;
        println!("  Warmup: {} outputs", warmup.len());
    }

    // Benchmark 100 iterations
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        let _outputs = session.run(ort::inputs![input_tensor.view()])?;
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / n;
    println!("  Average inference: {:.2} ms (over {} iterations)", avg_us as f64 / 1000.0, n);
    println!("  Throughput: {:.1} images/sec", 1_000_000.0 / avg_us as f64);

    // Memory check
    println!("  Note: Check Activity Monitor for unified memory growth");
    println!("  Run 500 iterations manually to verify no memory leak");

    Ok(())
}
