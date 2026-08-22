//! Hardware-accelerated image decoding via Apple ImageIO (macOS) or `image` crate (fallback).

#[cfg(target_os = "macos")]
mod platform {
    use objc2::rc::autoreleasepool;
    use objc2_core_foundation::{
        CFBoolean, CFData, CFDictionary, CFNumber, CFType, CGPoint, CGRect, CGSize,
    };
    use objc2_core_graphics::{
        CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
        CGImageByteOrderInfo,
    };
    use objc2_image_io::CGImageSource;

    /// Render a CGImage into an RGBA pixel buffer.
    fn cgimage_to_rgba(image: &CGImage) -> Result<(Vec<u8>, u32, u32), String> {
        let width = CGImage::width(Some(image)) as u32;
        let height = CGImage::height(Some(image)) as u32;

        if width == 0 || height == 0 {
            return Err("Image has zero dimensions".into());
        }

        let color_space = CGColorSpace::new_device_rgb()
            .ok_or_else(|| "Failed to create device RGB color space".to_string())?;

        // ByteOrder32Big | PremultipliedLast  → RGBA in natural byte order
        let bitmap_info =
            CGImageByteOrderInfo::Order32Big.0 | CGImageAlphaInfo::PremultipliedLast.0;

        let row_bytes = (width * 4) as usize;
        let buf_size = row_bytes * height as usize;
        let mut buffer: Vec<u8> = vec![0u8; buf_size];

        // SAFETY: CGBitmapContextCreate writes into the provided buffer.
        // The buffer is large enough (width * height * 4 bytes), the color
        // space is valid sRGB, and the bitmap info matches the 8-bit-per-
        // component RGBA layout.
        let ctx = unsafe {
            CGBitmapContextCreate(
                buffer.as_mut_ptr() as *mut core::ffi::c_void,
                width as usize,
                height as usize,
                8, // bits per component
                row_bytes,
                Some(&color_space),
                bitmap_info,
            )
        }
        .ok_or_else(|| "Failed to create bitmap context".to_string())?;

        let rect = CGRect {
            origin: CGPoint::new(0.0, 0.0),
            size: CGSize {
                width: width as f64,
                height: height as f64,
            },
        };
        CGContext::draw_image(Some(&ctx), rect, Some(image));

        Ok((buffer, width, height))
    }

    pub fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        autoreleasepool(|_| {
            let cf_data = CFData::from_bytes(bytes);

            let source = unsafe {
                CGImageSource::with_data(&cf_data, None)
                    .ok_or_else(|| "Failed to create image source".to_string())?
            };

            let cg_image = unsafe {
                source
                    .image_at_index(0, None)
                    .ok_or_else(|| "Failed to decode image at index 0".to_string())?
            };

            cgimage_to_rgba(&cg_image)
        })
    }

    pub fn decode_thumbnail(bytes: &[u8], max_px: i64) -> Result<(Vec<u8>, u32, u32), String> {
        autoreleasepool(|_| {
            let cf_data = CFData::from_bytes(bytes);

            let source = unsafe {
                CGImageSource::with_data(&cf_data, None)
                    .ok_or_else(|| "Failed to create image source".to_string())?
            };

            let max_px_num = CFNumber::new_i64(max_px);

            // SAFETY: accessing extern static CFString constants from the
            // ImageIO framework. These are always initialized on macOS.
            let keys: [&CFType; 4] = unsafe {
                [
                    objc2_image_io::kCGImageSourceCreateThumbnailFromImageAlways.as_ref(),
                    objc2_image_io::kCGImageSourceThumbnailMaxPixelSize.as_ref(),
                    objc2_image_io::kCGImageSourceCreateThumbnailWithTransform.as_ref(),
                    objc2_image_io::kCGImageSourceShouldCacheImmediately.as_ref(),
                ]
            };
            let values: [&CFType; 4] = [
                CFBoolean::new(true).as_ref(),
                max_px_num.as_ref(),
                CFBoolean::new(true).as_ref(),
                CFBoolean::new(true).as_ref(),
            ];

            let options = CFDictionary::<CFType, CFType>::from_slices(&keys, &values);

            let cg_image = unsafe {
                source
                    .thumbnail_at_index(0, Some(options.as_ref()))
                    .ok_or_else(|| "Failed to create thumbnail".to_string())?
            };

            cgimage_to_rgba(&cg_image)
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    pub fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Ok((rgba.into_raw(), w, h))
    }

    pub fn decode_thumbnail(bytes: &[u8], max_px: i64) -> Result<(Vec<u8>, u32, u32), String> {
        let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
        let (w, h) = img.dimensions();
        let longest = w.max(h) as i64;
        let img = if longest > max_px {
            let scale = max_px as f32 / longest as f32;
            img.resize(
                (w as f32 * scale) as u32,
                (h as f32 * scale) as u32,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            img
        };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Ok((rgba.into_raw(), w, h))
    }
}

/// Decode image to RGBA pixels at full resolution.
/// Returns (rgba_pixels, width, height).
pub fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    platform::decode_image(bytes)
}

/// Decode image to RGBA pixels, downscaled so longest dimension ≤ max_px.
/// Uses ImageIO thumbnail API which decodes+downscales in one pass (skips full-res decode).
/// Also applies EXIF orientation transform.
/// Returns (rgba_pixels, width, height).
pub fn decode_thumbnail(bytes: &[u8], max_px: i64) -> Result<(Vec<u8>, u32, u32), String> {
    platform::decode_thumbnail(bytes, max_px)
}
