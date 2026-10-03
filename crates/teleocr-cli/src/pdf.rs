//! PDF pages to images with PDFium (loaded at runtime: `PDFIUM_LIB_PATH`
//! directory, the executable's directory, then the system library path).
//! `scripts/fetch-pdfium.sh` downloads a build.

use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;
use pdfium_render::prelude::*;
use std::sync::{Mutex, OnceLock};

pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF")
}

/// PDFium binds once per process (a second bind fails with
/// `PdfiumLibraryBindingsAlreadyInitialized`), so the server shares one
/// instance across requests; a failed bind is retried on the next call.
fn pdfium() -> Result<&'static Pdfium> {
    static PDFIUM: OnceLock<Pdfium> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(p) = PDFIUM.get() {
        return Ok(p);
    }
    let _guard = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = PDFIUM.get() {
        return Ok(p);
    }
    let p = bind()?;
    Ok(PDFIUM.get_or_init(|| p))
}

fn bind() -> Result<Pdfium> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("PDFIUM_LIB_PATH") {
        dirs.push(std::path::PathBuf::from(d));
    }
    if let Some(d) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
    {
        dirs.push(d);
    }
    for d in &dirs {
        let lib = Pdfium::pdfium_platform_library_name_at_path(d);
        if lib.exists() {
            let b = Pdfium::bind_to_library(&lib).map_err(|e| anyhow!("loading {lib:?}: {e}"))?;
            return Ok(Pdfium::new(b));
        }
    }
    let b = Pdfium::bind_to_system_library().map_err(|e| {
        anyhow!("PDFium not found ({e}); run scripts/fetch-pdfium.sh or set PDFIUM_LIB_PATH")
    })?;
    Ok(Pdfium::new(b))
}

/// Parses "1-3,5" (1-based, inclusive) into 0-based indices. Ranges past the
/// last page are clipped ("1-8" on a 3-page PDF is pages 1-3); only a spec
/// that selects no page at all is an error.
pub fn parse_pages(spec: &str, n: usize) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<usize>()?, b.trim().parse::<usize>()?),
            None => {
                let p = part.parse::<usize>()?;
                (p, p)
            }
        };
        if a == 0 || b < a {
            bail!("invalid page range '{part}'");
        }
        out.extend(a - 1..b.min(n));
    }
    if out.is_empty() {
        bail!("page range '{spec}' outside 1-{n}");
    }
    Ok(out)
}

/// Renders the selected pages (all when `pages` is None) at `dpi`.
pub fn render(bytes: &[u8], dpi: f32, pages: Option<&str>) -> Result<Vec<RgbImage>> {
    let pdfium = pdfium()?;
    let doc = pdfium
        .load_pdf_from_byte_slice(bytes, None)
        .map_err(|e| anyhow!("reading PDF: {e}"))?;
    let n = doc.pages().len() as usize;
    let idx = match pages {
        Some(spec) => parse_pages(spec, n)?,
        None => (0..n).collect(),
    };
    let mut out = Vec::with_capacity(idx.len());
    for i in idx {
        let page = doc
            .pages()
            .get(i as _)
            .map_err(|e| anyhow!("page {}: {e}", i + 1))?;
        let scale = dpi / 72.0;
        let cfg = PdfRenderConfig::new().scale_page_by_factor(scale);
        let bmp = page
            .render_with_config(&cfg)
            .map_err(|e| anyhow!("rendering page {}: {e}", i + 1))?;
        let (w, h) = (bmp.width() as u32, bmp.height() as u32);
        let rgba = bmp.as_rgba_bytes();
        let rgb: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        out.push(RgbImage::from_raw(w, h, rgb).context("bitmap size")?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::parse_pages;

    // covers: REQ-PDF-001
    #[test]
    fn page_ranges() {
        assert_eq!(parse_pages("1-3,5", 5).unwrap(), vec![0, 1, 2, 4]);
        assert_eq!(parse_pages("1-8", 1).unwrap(), vec![0]);
        assert_eq!(parse_pages("2-8,9", 3).unwrap(), vec![1, 2]);
        assert!(parse_pages("5-8", 1).is_err());
        assert!(parse_pages("0-2", 3).is_err());
        assert!(parse_pages("3-1", 3).is_err());
    }
}
