//! Image preprocessing functions — ports of immich_ml/models/transforms.py

use image::RgbImage;

/// Decode an image from bytes, return RGBA pixels + dimensions.
pub fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    immich_ml_imaging::decode_image(bytes)
}

/// Letterbox: resize image to fit within (size × size) preserving aspect ratio,
/// pad the rest with black. Returns (canvas, scale) where scale = new_height / original_height.
/// Port of transforms.py::letterbox
///
/// Takes raw RGBA pixels + dimensions (from the imaging crate) instead of a DynamicImage.
pub fn letterbox(rgba: (&[u8], u32, u32), size: u32) -> (RgbImage, f32) {
    let (pixels, width, height) = rgba;

    // Convert RGBA → RGB (drop alpha channel)
    let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);
    for chunk in pixels.chunks_exact(4) {
        rgb_pixels.push(chunk[0]);
        rgb_pixels.push(chunk[1]);
        rgb_pixels.push(chunk[2]);
    }
    let rgb = RgbImage::from_raw(width, height, rgb_pixels)
        .expect("failed to create RgbImage from RGBA data");

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

/// Convert RGBA pixels + dimensions to an `image::RgbImage` (drops alpha).
pub fn rgba_to_rgb_image(rgba: &[u8], width: u32, height: u32) -> RgbImage {
    let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        rgb_pixels.push(chunk[0]);
        rgb_pixels.push(chunk[1]);
        rgb_pixels.push(chunk[2]);
    }
    RgbImage::from_raw(width, height, rgb_pixels)
        .expect("failed to create RgbImage from RGBA data")
}

/// Resize image if encoded size exceeds max_bytes, preserving aspect ratio.
/// Used for cloud API image transfer.
/// Uses the imaging crate for decoding+downscaling, then re-encodes as JPEG.
pub fn resize_if_too_large(bytes: &[u8], max_bytes: usize) -> Result<Vec<u8>, String> {
    if bytes.len() <= max_bytes {
        return Ok(bytes.to_vec());
    }

    // Decode + downscale in one pass using the imaging crate (max 1024px longest side).
    let (rgba, w, h) = immich_ml_imaging::decode_thumbnail(bytes, 1024)?;

    // Convert to RGB and re-encode as JPEG using the `image` crate.
    let rgb = rgba_to_rgb_image(&rgba, w, h);
    let dynamic = image::DynamicImage::ImageRgb8(rgb);

    let mut quality = 85u8;
    loop {
        let mut buf = std::io::Cursor::new(Vec::new());
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
        encoder.encode_image(&dynamic)
            .map_err(|e| format!("Image encode: {}", e))?;
        let encoded = buf.into_inner();
        if encoded.len() <= max_bytes || quality <= 20 {
            return Ok(encoded);
        }
        quality -= 10;
    }
}
