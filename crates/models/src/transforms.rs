//! Image preprocessing functions — ports of immich_ml/models/transforms.py

use image::{DynamicImage, GenericImageView, RgbImage};

/// Decode an image from bytes, convert to RGB.
pub fn decode_image(bytes: &[u8]) -> Result<DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|e| format!("Image decode: {}", e))
}

/// Letterbox: resize image to fit within (size × size) preserving aspect ratio,
/// pad the rest with black. Returns (canvas, scale) where scale = new_height / original_height.
/// Port of transforms.py::letterbox
pub fn letterbox(img: &DynamicImage, size: u32) -> (RgbImage, f32) {
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let (new_w, new_h) = if height > width {
        (size * width / height, size)
    } else {
        (size, size * height / width)
    };

    let resized = image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::Lanczos3);
    
    let mut canvas = RgbImage::new(size, size);
    image::imageops::overlay(&mut canvas, &resized, 0, 0);
    
    let scale = new_h as f32 / height as f32;
    (canvas, scale)
}

/// Normalize pixel values: (pixel / std) - (mean / std)
/// Port of transforms.py::normalize
/// Note: Python does `img *= 1/std; img -= mean/std` which equals `(img - mean) / std`
pub fn normalize(data: &mut [f32], mean: f32, std: f32) {
    let inv_std = 1.0 / std;
    let offset = mean / std;
    for v in data.iter_mut() {
        *v = *v * inv_std - offset;
    }
}

/// Resize image if encoded size exceeds max_bytes, preserving aspect ratio.
/// Used for cloud API image transfer.
pub fn resize_if_too_large(bytes: &[u8], max_bytes: usize) -> Result<Vec<u8>, String> {
    if bytes.len() <= max_bytes {
        return Ok(bytes.to_vec());
    }

    let img = image::load_from_memory(bytes).map_err(|e| format!("Image decode: {}", e))?;
    let mut current = img;
    let mut quality = 85u8;

    loop {
        let mut buf = std::io::Cursor::new(Vec::new());
        current.write_to(&mut buf, image::ImageFormat::Jpeg)
            .map_err(|e| format!("Image encode: {}", e))?;
        let encoded = buf.into_inner();
        if encoded.len() <= max_bytes || quality <= 20 {
            return Ok(encoded);
        }
        let new_w = (current.width() as f32 * 0.75) as u32;
        let new_h = (current.height() as f32 * 0.75) as u32;
        current = current.resize(new_w.max(1), new_h.max(1), image::imageops::FilterType::Lanczos3);
        quality -= 10;
    }
}
