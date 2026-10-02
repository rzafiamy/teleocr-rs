//! Full-page document parsing, after the official TeleOCR client
//! (`TeleOCR_client.py`): layout pass on a 1036x1036 copy of the page →
//! crop every block from the full-resolution page → per-type recognition
//! prompt → post-processing → Markdown in the layout's reading order.

use crate::model::{Engine, GenerateOptions, Output};
use crate::otsl;
use anyhow::Result;
use image::{Rgb, RgbImage};
use regex::Regex;
use serde::Serialize;
use std::sync::LazyLock;

pub const LAYOUT_SIZE: u32 = 1036;
const MIN_IMAGE_EDGE: u32 = 28;
const MAX_IMAGE_EDGE_RATIO: f32 = 50.0;
/// Pages are downscaled above this many pixels before cropping.
const MAX_PAGE_PIXELS: u64 = 8000 * 8000;

pub const BLOCK_TYPES: &[&str] = &[
    "text",
    "title",
    "table",
    "image",
    "code",
    "algorithm",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "aside_text",
    "equation",
    "equation_block",
    "ref_text",
    "list",
    "phonetic",
    "table_caption",
    "image_caption",
    "code_caption",
    "table_footnote",
    "image_footnote",
    "unknown",
    "seal",
    "char",
];

/// Blocks that are not sent to recognition (containers and pictures).
const SKIP_EXTRACT: &[&str] = &["image", "list", "equation_block"];
/// Page furniture, left out of the Markdown.
const PARATEXT: &[&str] = &[
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LayoutMode {
    /// Axis-aligned boxes; clean digital pages.
    Detection,
    /// Polygons; photographed, curved or skewed pages.
    Segmentation,
}

impl LayoutMode {
    fn prompt(self) -> &'static str {
        match self {
            LayoutMode::Detection => "\nAnalyze the image layout.",
            LayoutMode::Segmentation => "\nMulti-point Layout Segmentation Analysis.",
        }
    }
}

/// Recognition prompt for a block type (`DEFAULT_PROMPTS`).
pub fn block_prompt(kind: &str) -> &'static str {
    match kind {
        "table" => "\nThis is the image of a table. Please output the table in OTSL format.",
        "equation" => {
            "\nPlease write out the expression of the formula in the image using LaTeX format."
        }
        "code" => "\nThe image contains a code snippet, please output the parsing result.",
        "seal" => "\nSeal Recognition:",
        "char" => "\nThis is a scientific figure. Please extract the table implied by this figure.",
        _ => "\nPlease output the text content from the image.",
    }
}

fn sampling_kind(kind: &str) -> &str {
    match kind {
        "text" | "table" | "equation" | "code" | "char" | "seal" => kind,
        _ => "default",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Block {
    #[serde(rename = "type")]
    pub kind: String,
    /// `[x0, y0, x1, y1]` as fractions of the page size.
    pub bbox: [f32; 4],
    /// Polygon `[x, y, ...]` (fractions) in segmentation mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<f32>>,
    /// Text orientation: 0, 90, 180 or 270 degrees.
    pub angle: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub mode: LayoutMode,
    /// Keep headers, footers, page numbers … in the Markdown.
    pub paratext: bool,
    pub max_new_tokens: usize,
    /// Blocks decoded together (1 = one at a time).
    pub batch: usize,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            mode: LayoutMode::Detection,
            paratext: false,
            max_new_tokens: 4096,
            batch: 8,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PageStats {
    pub layout_ms: f64,
    pub extract_ms: f64,
    pub blocks: usize,
    pub generated_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Page {
    pub width: u32,
    pub height: u32,
    pub blocks: Vec<Block>,
    pub markdown: String,
    pub stats: PageStats,
}

static LAYOUT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$").unwrap());

/// Parses the layout pass output into blocks (reading order preserved).
pub fn parse_layout(output: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        let Some(m) = LAYOUT_RE.captures(line) else {
            if !line.is_empty() {
                tracing::warn!("layout line ignored: {line}");
            }
            continue;
        };
        let kind = m[2].to_lowercase();
        if !BLOCK_TYPES.contains(&kind.as_str()) {
            tracing::warn!("unknown block type: {kind}");
            continue;
        }
        let nums: Vec<i64> = m[1]
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if nums.len() < 4
            || !nums.len().is_multiple_of(2)
            || nums.iter().any(|&n| !(0..=1000).contains(&n))
        {
            tracing::warn!("bad box: {line}");
            continue;
        }
        let f: Vec<f32> = nums.iter().map(|&n| n as f32 / 1000.0).collect();
        let xs = f.iter().step_by(2);
        let ys = f.iter().skip(1).step_by(2);
        let bbox = [
            xs.clone().cloned().fold(f32::MAX, f32::min),
            ys.clone().cloned().fold(f32::MAX, f32::min),
            xs.cloned().fold(f32::MIN, f32::max),
            ys.cloned().fold(f32::MIN, f32::max),
        ];
        let tail = &m[3];
        let angle = [("up", 0), ("right", 90), ("down", 180), ("left", 270)]
            .iter()
            .find(|(t, _)| tail.contains(t))
            .map(|(_, a)| *a);
        blocks.push(Block {
            kind,
            bbox,
            polygon: (f.len() > 4).then_some(f),
            angle,
            content: None,
            lang: None,
        });
    }
    blocks
}

fn point_in_polygon(x: f32, y: f32, pts: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let (xi, yi) = pts[i];
        let (xj, yj) = pts[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Crops a block (pixels outside a polygon are blacked out, as cv2's
/// fillPoly + bitwise_and), rotates it upright and pads/upscales it.
fn crop_block(page: &RgbImage, b: &Block) -> Option<RgbImage> {
    let (w, h) = (page.width() as f32, page.height() as f32);
    let x0 = (b.bbox[0] * w) as u32;
    let y0 = (b.bbox[1] * h) as u32;
    let x1 = ((b.bbox[2] * w) as u32).min(page.width());
    let y1 = ((b.bbox[3] * h) as u32).min(page.height());
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut crop = image::imageops::crop_imm(page, x0, y0, x1 - x0, y1 - y0).to_image();
    if let Some(poly) = &b.polygon {
        let pts: Vec<(f32, f32)> = poly
            .chunks(2)
            .map(|p| {
                (
                    (p[0] * w).trunc() - x0 as f32,
                    (p[1] * h).trunc() - y0 as f32,
                )
            })
            .collect();
        for (x, y, px) in crop.enumerate_pixels_mut() {
            if !point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, &pts) {
                *px = Rgb([0, 0, 0]);
            }
        }
    }
    // PIL's rotate(angle) is counter-clockwise.
    crop = match b.angle {
        Some(90) => image::imageops::rotate270(&crop),
        Some(180) => image::imageops::rotate180(&crop),
        Some(270) => image::imageops::rotate90(&crop),
        _ => crop,
    };
    Some(resize_by_need(crop))
}

fn resize_by_need(mut img: RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let ratio = w.max(h) as f32 / w.min(h) as f32;
    if ratio > MAX_IMAGE_EDGE_RATIO {
        let (nw, nh) = if w > h {
            (w, (w as f32 / MAX_IMAGE_EDGE_RATIO).ceil() as u32)
        } else {
            ((h as f32 / MAX_IMAGE_EDGE_RATIO).ceil() as u32, h)
        };
        let mut canvas = RgbImage::from_pixel(nw, nh, Rgb([255, 255, 255]));
        image::imageops::overlay(
            &mut canvas,
            &img,
            ((nw - w) / 2) as i64,
            ((nh - h) / 2) as i64,
        );
        img = canvas;
    }
    let min = img.width().min(img.height());
    if min < MIN_IMAGE_EDGE {
        let s = MIN_IMAGE_EDGE as f32 / min as f32;
        let nw = (img.width() as f32 * s).ceil() as usize;
        let nh = (img.height() as f32 * s).ceil() as usize;
        img = crate::image::resize_bicubic(&img, nw, nh);
    }
    img
}

// ---------------------------------------------------------------- post-process

static MATH_OPEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\$\$)?\s*\\[\[(]\s*").unwrap());
static MATH_CLOSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*\\[\])]\s*(\$\$)?$").unwrap());
static TAIL_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\$\$(.*?)\s*(\([A-Za-z0-9][A-Za-z0-9.\-]*\))(\s*\$\$\s*)$").unwrap()
});
static TAIL_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(^|[^\\])tag\s*\{([^{}]*)\}(\s*\$\$\s*)$").unwrap());
static INLINE_DOLLAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\$").unwrap());

/// Display equation: drop `\[ \]` / `\( \)`, wrap in `$$`, move a trailing
/// equation number out, fix a bare `tag{}`.
pub fn post_equation(s: &str) -> String {
    let s = MATH_OPEN.replace(s, "$1");
    let s = MATH_CLOSE.replace(&s, "$1");
    let body = s.replace('$', "");
    let wrapped = format!("$${}$$", body.trim());
    if let Some(m) = TAIL_LABEL.captures(&wrapped)
        && !m[1].trim_end().ends_with("\\eqno")
    {
        return format!("${}$ {}", m[1].trim_end(), &m[2]);
    }
    TAIL_TAG.replace(&wrapped, "$1\\tag{$2}$3").into_owned()
}

/// Text: `$$…$$` → `$…$`, then space-pad balanced inline math.
pub fn post_text(s: &str) -> String {
    let text = s.replace("$$", "$");
    // Unescaped single dollars.
    let bytes = text.as_bytes();
    let ms: Vec<usize> = INLINE_DOLLAR
        .find_iter(&text)
        .map(|m| m.start())
        .filter(|&i| i == 0 || bytes[i - 1] != b'\\')
        .collect();
    if ms.is_empty() || ms.len() % 2 == 1 {
        return text;
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut last = 0;
    for pair in ms.chunks(2) {
        let (l, r) = (pair[0], pair[1]);
        out.push_str(&text[last..l]);
        if l > 0 && !text[..l].ends_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push('$');
        out.push_str(text[l + 1..r].trim());
        out.push('$');
        if r + 1 < text.len() && !text[r + 1..].starts_with(char::is_whitespace) {
            out.push(' ');
        }
        last = r + 1;
    }
    out.push_str(&text[last..]);
    out
}

pub fn post_table(s: &str) -> String {
    otsl::to_html(s)
        .replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
        .replace("<thead>", "")
        .replace("</thead>", "")
        .replace("<table>", "<table border=\"1\">")
}

/// Strips a ``` fence and a `<_Lang_>` prefix; returns (lang, code).
pub fn split_code(s: &str) -> (String, String) {
    let lines: Vec<&str> = s.lines().collect();
    let mut a = 0;
    let mut b = lines.len();
    if lines.first().is_some_and(|l| l.starts_with("```")) {
        a = 1;
    }
    if b > a && lines[b - 1].trim() == "```" {
        b -= 1;
    }
    let code = lines[a.min(b)..b].join("\n").trim().to_string();
    if let Some(rest) = code.strip_prefix("<_")
        && let Some(end) = rest.find("_>")
    {
        return (rest[..end].to_lowercase(), rest[end + 2..].to_string());
    }
    ("txt".into(), code)
}

fn post_process(b: &mut Block) {
    let Some(c) = b.content.take() else { return };
    let c = match b.kind.as_str() {
        "equation" => post_equation(&c),
        "text" => post_text(&c),
        "table" => post_table(&c),
        "char" if c.contains("<fcel>") || c.contains("<ecel>") => post_table(&c),
        "code" | "algorithm" => {
            let (lang, code) = split_code(&c);
            b.lang = Some(lang);
            code
        }
        "title" => c.split_whitespace().collect::<Vec<_>>().join(" "),
        _ => c,
    };
    b.content = Some(c);
}

/// Markdown in reading order (`mk_blocks_to_markdown`, simplified: blocks
/// are emitted one by one; captions stay next to their table/figure
/// because the layout lists them in reading order).
pub fn to_markdown(blocks: &[Block], paratext: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        if !paratext && PARATEXT.contains(&b.kind.as_str()) {
            continue;
        }
        let Some(c) = b.content.as_deref() else {
            continue;
        };
        let c = c.trim();
        if c.is_empty() {
            continue;
        }
        parts.push(match b.kind.as_str() {
            "title" => format!("# {c}"),
            "code" => format!("```{}\n{c}\n```", b.lang.as_deref().unwrap_or("txt")),
            "char" if c.len() <= 5 => continue,
            _ => c.to_string(),
        });
    }
    parts.join("\n\n")
}

impl Engine {
    /// Parses one page image into blocks and Markdown.
    pub fn parse_page(&self, page: &RgbImage, opts: &ParseOptions) -> Result<Page> {
        Ok(self
            .parse_pages(std::slice::from_ref(page), opts)?
            .remove(0))
    }

    /// Parses several pages: the layout passes of all pages run as one
    /// batch, then the blocks of all pages. Stage timings are per batch.
    pub fn parse_pages(&self, pages: &[RgbImage], opts: &ParseOptions) -> Result<Vec<Page>> {
        let t0 = std::time::Instant::now();
        let mut lopts = GenerateOptions::pipeline("layout");
        lopts.max_new_tokens = opts.max_new_tokens;
        let layout_imgs: Vec<RgbImage> = pages
            .iter()
            .map(|p| crate::image::resize_bicubic(p, LAYOUT_SIZE as usize, LAYOUT_SIZE as usize))
            .collect();
        let jobs: Vec<_> = layout_imgs
            .iter()
            .map(|i| (i, opts.mode.prompt(), &lopts))
            .collect();
        let layouts: Vec<Output> = self.generate_batch(&jobs, opts.batch)?;
        let layout_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = std::time::Instant::now();
        let mut all_blocks: Vec<Vec<Block>> =
            layouts.iter().map(|l| parse_layout(&l.text)).collect();
        let mut stats: Vec<PageStats> = all_blocks
            .iter()
            .zip(&layouts)
            .map(|(b, l)| PageStats {
                layout_ms,
                blocks: b.len(),
                generated_tokens: l.tokens.len(),
                ..Default::default()
            })
            .collect();

        // (page, block, crop, options) for every block to recognize.
        let mut crops = Vec::new();
        for (pi, (page, blocks)) in pages.iter().zip(&all_blocks).enumerate() {
            let px = page.width() as u64 * page.height() as u64;
            let scaled;
            let src = if px > MAX_PAGE_PIXELS {
                let s = (MAX_PAGE_PIXELS as f64 / px as f64).sqrt();
                let (w, h) = (
                    (page.width() as f64 * s).round() as usize,
                    (page.height() as f64 * s).round() as usize,
                );
                scaled = crate::image::resize_bicubic(page, w.max(1), h.max(1));
                &scaled
            } else {
                page
            };
            for (bi, b) in blocks.iter().enumerate() {
                if SKIP_EXTRACT.contains(&b.kind.as_str()) {
                    continue;
                }
                let Some(crop) = crop_block(src, b) else {
                    continue;
                };
                let mut gopts = GenerateOptions::pipeline(sampling_kind(&b.kind));
                gopts.max_new_tokens = opts.max_new_tokens;
                crops.push((pi, bi, crop, gopts));
            }
        }
        let jobs: Vec<_> = crops
            .iter()
            .map(|(pi, bi, crop, o)| (crop, block_prompt(&all_blocks[*pi][*bi].kind), o))
            .collect();
        let outs = self.generate_batch(&jobs, opts.batch)?;
        for ((pi, bi, _, _), out) in crops.iter().zip(outs) {
            let b = &mut all_blocks[*pi][*bi];
            stats[*pi].generated_tokens += out.tokens.len();
            if out.truncated {
                tracing::warn!("{} block hit max_new_tokens", b.kind);
            }
            b.content = Some(out.text);
            post_process(b);
        }
        let extract_ms = t0.elapsed().as_secs_f64() * 1e3;

        Ok(pages
            .iter()
            .zip(all_blocks)
            .zip(stats)
            .map(|((page, blocks), mut stats)| {
                stats.extract_ms = extract_ms;
                Page {
                    width: page.width(),
                    height: page.height(),
                    markdown: to_markdown(&blocks, opts.paratext),
                    blocks,
                    stats,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_lines() {
        let b = parse_layout(
            "<box:796 035 910 067><label:header><up>\n<box:088 077 490 105><label:equation><left>\nbogus",
        );
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].kind, "header");
        assert_eq!(b[0].bbox, [0.796, 0.035, 0.91, 0.067]);
        assert_eq!(b[1].angle, Some(270));
        let p = parse_layout("<box:10 10 100 12 98 90 12 95><label:text><up>");
        assert_eq!(p[0].bbox, [0.01, 0.01, 0.1, 0.095]);
        assert!(p[0].polygon.is_some());
    }

    #[test]
    fn equations() {
        assert_eq!(post_equation("\\[ E=mc^2 \\]"), "$$E=mc^2$$");
        assert_eq!(post_equation("$$E=mc^2(1-2)$$"), "$E=mc^2$ (1-2)");
        assert_eq!(
            post_equation("$$E=mc^2\\eqno(1-2)$$"),
            "$$E=mc^2\\eqno(1-2)$$"
        );
        assert_eq!(post_equation("x tag{3}"), "$$x \\tag{3}$$");
    }

    #[test]
    fn inline_math() {
        assert_eq!(post_text("a$x$b"), "a $x$ b");
        assert_eq!(post_text("cost $$ 5"), "cost $ 5");
        assert_eq!(post_text("a $ x $ b"), "a $x$ b");
    }

    #[test]
    fn code() {
        assert_eq!(
            split_code("<_JavaScript_>let a;"),
            ("javascript".into(), "let a;".into())
        );
        assert_eq!(split_code("```\nx\n```"), ("txt".into(), "x".into()));
    }
}
