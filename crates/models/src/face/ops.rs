//! SCRFD decode, NMS, Umeyama similarity transform, face alignment.
//! Ports of immich_ml/models/facial_recognition/_ops.py

use image::RgbImage;

pub const DET_SIZE: u32 = 640;
pub const ALIGNED_SIZE: u32 = 112;
const DET_STRIDES: [u32; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;

/// Canonical ArcFace 5-point template for a 112×112 crop.
const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Generate anchor centers for a given feature map size and stride.
/// Port of _ops.py::_anchor_centers
fn anchor_centers(size: u32, stride: u32) -> Vec<[f32; 2]> {
    let grid = size / stride;
    let mut centers = Vec::with_capacity((grid * grid * ANCHORS_PER_CELL as u32) as usize);
    for y in 0..grid {
        for x in 0..grid {
            let cx = x as f32 * stride as f32;
            let cy = y as f32 * stride as f32;
            for _ in 0..ANCHORS_PER_CELL {
                centers.push([cx, cy]);
            }
        }
    }
    centers
}

/// Decode SCRFD model output heads into (scores, boxes, keypoints).
/// Port of _ops.py::decode_scrfd
///
/// `heads` is a list of 9 arrays in order:
///   [0-2] scores at strides 8, 16, 32
///   [3-5] box distances at strides 8, 16, 32
///   [6-8] keypoint offsets at strides 8, 16, 32
pub fn decode_scrfd(heads: &[Vec<f32>], _head_shapes: &[(usize, usize, usize)]) -> (Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 10]>) {
    let mut all_scores = Vec::new();
    let mut all_boxes = Vec::new();
    let mut all_kps = Vec::new();

    for (level, &stride) in DET_STRIDES.iter().enumerate() {
        let centers = anchor_centers(DET_SIZE, stride);
        let num_anchors = centers.len();

        // Score head: shape [1, num_anchors, 1] or [1, num_anchors]
        let score_head = &heads[level];
        let box_head = &heads[level + DET_STRIDES.len()];
        let kps_head = &heads[level + 2 * DET_STRIDES.len()];

        for i in 0..num_anchors {
            // Score
            all_scores.push(score_head[i]);

            // Box: [left, top, right, bottom] distances from center
            let dl = box_head[i * 4] * stride as f32;
            let dt = box_head[i * 4 + 1] * stride as f32;
            let dr = box_head[i * 4 + 2] * stride as f32;
            let db = box_head[i * 4 + 3] * stride as f32;
            all_boxes.push([
                centers[i][0] - dl, // x1
                centers[i][1] - dt, // y1
                centers[i][0] + dr, // x2
                centers[i][1] + db, // y2
            ]);

            // Keypoints: 5 points × 2 offsets (dx, dy from center)
            let mut kps = [0.0f32; 10];
            for k in 0..5 {
                kps[k * 2] = centers[i][0] + kps_head[i * 10 + k * 2] * stride as f32;
                kps[k * 2 + 1] = centers[i][1] + kps_head[i * 10 + k * 2 + 1] * stride as f32;
            }
            all_kps.push(kps);
        }
    }

    (all_scores, all_boxes, all_kps)
}

/// Non-maximum suppression. Returns indices of boxes to keep.
/// Port of _ops.py::nms (uses cv2.dnn.NMSBoxes in Python, reimplemented here)
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], threshold: f32) -> Vec<usize> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort indices by score descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; n];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if j == i || suppressed[j] {
                continue;
            }
            if iou(&boxes[i], &boxes[j]) > threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Intersection over Union for two boxes [x1, y1, x2, y2]
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);

    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Umeyama similarity transform: compute affine matrix mapping src points to dst points.
/// Returns a 2×3 matrix [row0, row1] where each row is [a, b, tx].
/// Port of _ops.py::umeyama
pub fn umeyama(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [[f32; 3]; 2] {
    let n = 5.0f32;

    // Compute means
    let src_mean = mean_2d(src);
    let dst_mean = mean_2d(dst);

    // Centered points
    let src_c: Vec<[f32; 2]> = src.iter().map(|p| [p[0] - src_mean[0], p[1] - src_mean[1]]).collect();
    let dst_c: Vec<[f32; 2]> = dst.iter().map(|p| [p[0] - dst_mean[0], p[1] - dst_mean[1]]).collect();

    // Covariance matrix C = dst_c^T * src_c / n  (2×2)
    let mut cov = [[0.0f32; 2]; 2];
    for i in 0..5 {
        cov[0][0] += dst_c[i][0] * src_c[i][0];
        cov[0][1] += dst_c[i][0] * src_c[i][1];
        cov[1][0] += dst_c[i][1] * src_c[i][0];
        cov[1][1] += dst_c[i][1] * src_c[i][1];
    }
    cov[0][0] /= n;
    cov[0][1] /= n;
    cov[1][0] /= n;
    cov[1][1] /= n;

    // SVD of 2×2 covariance matrix
    let (u, sigma, vt) = svd_2x2(cov);

    // d = sign(det(U * V^T))
    let det_uvt = u[0][0] * vt[0][0] + u[0][1] * vt[1][0]; // det of U * V^T (both rotation matrices)
    let d = if det_uvt < 0.0 { -1.0 } else { 1.0 };

    // R = U * diag(1, d) * V^T
    let diag = [[1.0f32, 0.0], [0.0, d]];
    // R = U * diag * V^T
    let ud = mat_mul_2x2(&u, &diag);
    let r = mat_mul_2x2(&ud, &vt);

    // scale = trace(diag * Σ) / sum(src_c²) * n
    // trace(diag * Σ) = 1*sigma[0] + d*sigma[1]
    let src_var: f32 = src_c.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum();
    let scale = (sigma[0] + d * sigma[1]) / src_var * n;

    // translation = dst_mean - scale * R * src_mean
    let rs = mat_vec_2x2(&r, &src_mean);
    let tx = dst_mean[0] - scale * rs[0];
    let ty = dst_mean[1] - scale * rs[1];

    // Affine matrix [scale*R | t] (2×3)
    [
        [scale * r[0][0], scale * r[0][1], tx],
        [scale * r[1][0], scale * r[1][1], ty],
    ]
}

/// Align face: warp image using Umeyama transform to produce 112×112 crop.
/// Port of _ops.py::align_face (uses cv2.warpAffine in Python)
pub fn align_face(image: &RgbImage, kps: &[f32; 10]) -> RgbImage {
    // Extract 5 keypoints from flat array
    let src_pts: [[f32; 2]; 5] = [
        [kps[0], kps[1]],
        [kps[2], kps[3]],
        [kps[4], kps[5]],
        [kps[6], kps[7]],
        [kps[8], kps[9]],
    ];

    let matrix = umeyama(&src_pts, &ARCFACE_DST);
    warp_affine(image, &matrix, ALIGNED_SIZE, ALIGNED_SIZE)
}

/// Apply affine warp using backward mapping with bilinear interpolation.
/// matrix is 2×3: [[a, b, tx], [c, d, ty]]
/// dst(x, y) = src(a*x + b*y + tx, c*x + d*y + ty)
fn warp_affine(image: &RgbImage, matrix: &[[f32; 3]; 2], out_w: u32, out_h: u32) -> RgbImage {
    let (src_w, src_h) = image.dimensions();
    let mut output = RgbImage::new(out_w, out_h);

    let a = matrix[0][0];
    let b = matrix[0][1];
    let tx = matrix[0][2];
    let c = matrix[1][0];
    let d = matrix[1][1];
    let ty = matrix[1][2];

    for y in 0..out_h {
        for x in 0..out_w {
            let sx = a * x as f32 + b * y as f32 + tx;
            let sy = c * x as f32 + d * y as f32 + ty;
            
            let pixel = bilinear_sample(image, sx, sy, src_w, src_h);
            output.put_pixel(x, y, pixel);
        }
    }
    output
}

/// Bilinear interpolation sampling at fractional coordinates.
fn bilinear_sample(image: &RgbImage, x: f32, y: f32, w: u32, h: u32) -> image::Rgb<u8> {
    if x < 0.0 || y < 0.0 || x >= w as f32 - 1.0 || y >= h as f32 - 1.0 {
        return image::Rgb([0, 0, 0]);
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);

    let r = (1.0 - dx) * (1.0 - dy) * p00[0] as f32
        + dx * (1.0 - dy) * p10[0] as f32
        + (1.0 - dx) * dy * p01[0] as f32
        + dx * dy * p11[0] as f32;
    let g = (1.0 - dx) * (1.0 - dy) * p00[1] as f32
        + dx * (1.0 - dy) * p10[1] as f32
        + (1.0 - dx) * dy * p01[1] as f32
        + dx * dy * p11[1] as f32;
    let bl = (1.0 - dx) * (1.0 - dy) * p00[2] as f32
        + dx * (1.0 - dy) * p10[2] as f32
        + (1.0 - dx) * dy * p01[2] as f32
        + dx * dy * p11[2] as f32;

    image::Rgb([r.round() as u8, g.round() as u8, bl.round() as u8])
}

// --- Math helpers ---

fn mean_2d(pts: &[[f32; 2]; 5]) -> [f32; 2] {
    let mx: f32 = pts.iter().map(|p| p[0]).sum::<f32>() / 5.0;
    let my: f32 = pts.iter().map(|p| p[1]).sum::<f32>() / 5.0;
    [mx, my]
}

fn mat_mul_2x2(a: &[[f32; 2]; 2], b: &[[f32; 2]; 2]) -> [[f32; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

fn mat_vec_2x2(m: &[[f32; 2]; 2], v: &[f32; 2]) -> [f32; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

/// SVD of a 2×2 matrix using analytic eigenvalue decomposition.
/// Returns (U, [sigma1, sigma2], V^T) where A = U * diag(sigmas) * V^T
fn svd_2x2(a: [[f32; 2]; 2]) -> ([[f32; 2]; 2], [f32; 2], [[f32; 2]; 2]) {
    // Compute A^T * A (symmetric 2×2)
    let ata = [
        [a[0][0] * a[0][0] + a[1][0] * a[1][0], a[0][0] * a[0][1] + a[1][0] * a[1][1]],
        [a[0][0] * a[0][1] + a[1][0] * a[1][1], a[0][1] * a[0][1] + a[1][1] * a[1][1]],
    ];

    let p = ata[0][0];
    let q = ata[0][1];
    let r = ata[1][1];

    let trace = p + r;
    let det = p * r - q * q;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let lambda1 = (trace + disc) / 2.0;
    let lambda2 = (trace - disc) / 2.0;

    let sigma1 = lambda1.max(0.0).sqrt();
    let sigma2 = lambda2.max(0.0).sqrt();

    // Eigenvectors of A^T * A = V
    let v: [[f32; 2]; 2] = if q.abs() > 1e-10 {
        let v1 = [q, lambda1 - p];
        let v1_norm = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let v1 = [v1[0] / v1_norm, v1[1] / v1_norm];
        let v2 = [-v1[1], v1[0]]; // perpendicular
        [[v1[0], v2[0]], [v1[1], v2[1]]] // columns are v1, v2
    } else {
        // q ≈ 0: A^T*A is already diagonal
        if p >= r { [[1.0, 0.0], [0.0, 1.0]] } else { [[0.0, 1.0], [1.0, 0.0]] }
    };

    // U = A * V * Σ^-1 (if sigma_i > 0)
    let av = mat_mul_2x2(&a, &v); // A * V
    let u: [[f32; 2]; 2] = if sigma1 > 1e-10 && sigma2 > 1e-10 {
        [[av[0][0] / sigma1, av[0][1] / sigma2], [av[1][0] / sigma1, av[1][1] / sigma2]]
    } else if sigma1 > 1e-10 {
        let u0 = [av[0][0] / sigma1, av[1][0] / sigma1];
        let u0_norm = (u0[0] * u0[0] + u0[1] * u0[1]).sqrt();
        let u0 = [u0[0] / u0_norm, u0[1] / u0_norm];
        let u1 = [-u0[1], u0[0]];
        [[u0[0], u1[0]], [u0[1], u1[1]]]
    } else {
        [[1.0, 0.0], [0.0, 1.0]] // identity fallback
    };

    // V^T is transpose of V
    let vt = [[v[0][0], v[1][0]], [v[0][1], v[1][1]]];

    (u, [sigma1, sigma2], vt)
}
