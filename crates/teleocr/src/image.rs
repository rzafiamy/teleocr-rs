//! Image preprocessing of `Qwen2VLImageProcessor`: smart resize to a
//! multiple of 28 px, PIL-style antialiased bicubic resampling, CLIP
//! normalization, and the patch flattening the vision tower expects.

use anyhow::Result;
use image::RgbImage;

pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 3136,
            max_pixels: 12_845_056,
        }
    }
}

/// Patches of one image, ready for the vision tower.
pub struct Patches {
    /// `[grid_h * grid_w, C * T * P * P]`, row-major.
    pub data: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub dim: usize,
}

impl Patches {
    /// Number of LLM tokens after the 2x2 merge.
    pub fn n_tokens(&self, merge: usize) -> usize {
        self.grid_h * self.grid_w / (merge * merge)
    }
}

/// Python's `round()`: half to even.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `smart_resize` from transformers' Qwen2-VL image processor.
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_px: usize,
    max_px: usize,
) -> (usize, usize) {
    let (hf, wf, f) = (h as f64, w as f64, factor as f64);
    let mut hb = (round_half_even(hf / f) * f).max(f);
    let mut wb = (round_half_even(wf / f) * f).max(f);
    if hb * wb > max_px as f64 {
        let beta = (hf * wf / max_px as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min_px as f64 {
        let beta = (min_px as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    (hb as usize, wb as usize)
}

fn bicubic(x: f64) -> f64 {
    // a = -0.5, as PIL and torchvision's antialiased path.
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per output pixel: first input index and normalized weights.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f64 / out_size as f64;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    (0..out_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor() as usize).min(in_size);
            let mut ws: Vec<f64> = (xmin..xmax)
                .map(|x| bicubic((x as f64 - center + 0.5) / fscale))
                .collect();
            let total: f64 = ws.iter().sum();
            if total != 0.0 {
                ws.iter_mut().for_each(|w| *w /= total);
            }
            (xmin, ws.into_iter().map(|w| w as f32).collect())
        })
        .collect()
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Antialiased bicubic resize (horizontal pass, then vertical), 8-bit
/// intermediate like PIL.
pub fn resize_bicubic(img: &RgbImage, out_w: usize, out_h: usize) -> RgbImage {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    if in_w == out_w && in_h == out_h {
        return img.clone();
    }
    let src = img.as_raw();
    let cx = coeffs(in_w, out_w);
    let mut tmp = vec![0u8; out_w * in_h * 3];
    for y in 0..in_h {
        let row = &src[y * in_w * 3..(y + 1) * in_w * 3];
        for (x, (x0, ws)) in cx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, w) in ws.iter().enumerate() {
                let p = &row[(x0 + k) * 3..(x0 + k) * 3 + 3];
                for c in 0..3 {
                    acc[c] += p[c] as f32 * w;
                }
            }
            let o = (y * out_w + x) * 3;
            for c in 0..3 {
                tmp[o + c] = clamp_u8(acc[c]);
            }
        }
    }
    let cy = coeffs(in_h, out_h);
    let mut out = vec![0u8; out_w * out_h * 3];
    for (y, (y0, ws)) in cy.iter().enumerate() {
        for x in 0..out_w {
            let mut acc = [0f32; 3];
            for (k, w) in ws.iter().enumerate() {
                let o = ((y0 + k) * out_w + x) * 3;
                for c in 0..3 {
                    acc[c] += tmp[o + c] as f32 * w;
                }
            }
            let o = (y * out_w + x) * 3;
            for c in 0..3 {
                out[o + c] = clamp_u8(acc[c]);
            }
        }
    }
    RgbImage::from_raw(out_w as u32, out_h as u32, out).expect("buffer size")
}

/// Resize, normalize and flatten one image into vision patches.
pub fn preprocess(img: &RgbImage, cfg: &PreprocessConfig) -> Result<Patches> {
    let factor = cfg.patch_size * cfg.merge_size;
    let (h, w) = smart_resize(
        img.height() as usize,
        img.width() as usize,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    );
    let img = resize_bicubic(img, w, h);
    let raw = img.as_raw();

    let (p, m, t) = (cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size);
    let (gh, gw) = (h / p, w / p);
    let dim = 3 * t * p * p;
    let mut data = vec![0f32; gh * gw * dim];

    // Output order: merge blocks row-major, then the 2x2 patches inside a
    // block; inside a patch: channel, frame (the image repeated), row, col.
    let mut idx = 0;
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for sh in 0..m {
                for sw in 0..m {
                    let (py0, px0) = ((bh * m + sh) * p, (bw * m + sw) * p);
                    let base = idx * dim;
                    for c in 0..3 {
                        for f in 0..t {
                            for yy in 0..p {
                                for xx in 0..p {
                                    let v = raw[((py0 + yy) * w + px0 + xx) * 3 + c] as f32 / 255.0;
                                    data[base + ((c * t + f) * p + yy) * p + xx] =
                                        (v - MEAN[c]) / STD[c];
                                }
                            }
                        }
                    }
                    idx += 1;
                }
            }
        }
    }
    Ok(Patches {
        data,
        grid_h: gh,
        grid_w: gw,
        dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_python() {
        // Python: smart_resize(1036, 1036) == (1036, 1036)
        assert_eq!(smart_resize(1036, 1036, 28, 3136, 12_845_056), (1036, 1036));
        // round(70/28 = 2.5) == 2 in Python (half to even).
        assert_eq!(smart_resize(70, 70, 28, 0, 12_845_056), (56, 56));
        assert_eq!(smart_resize(10, 10, 28, 3136, 12_845_056), (56, 56));
    }
}
