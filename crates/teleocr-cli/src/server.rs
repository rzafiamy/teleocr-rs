//! HTTP API.
//!
//! - `POST /v1/ocr`: one image (JSON with base64 / data URI, or multipart
//!   field `file`); `task=parse` (default) runs the full document pipeline
//!   and returns Markdown + blocks, any other task (text, table, formula,
//!   code, layout, layout_seg, figure, seal) runs a single prompt.
//! - `POST /v1/chat/completions`: OpenAI-compatible, one `image_url`
//!   (data URI) plus text; the text is the prompt, or a task name.
//! - `GET /health`, `GET /v1/models`.
//!
//! The model runs one request at a time (requests queue on a mutex).

use anyhow::{Context, Result, anyhow};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use teleocr::pipeline::{self, LayoutMode, ParseOptions};
use teleocr::{Engine, GenerateOptions};
use tokio::sync::Mutex;

pub struct AppState {
    engine: Mutex<Engine>,
    model_id: String,
    batch: usize,
}

type Shared = Arc<AppState>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"message": self.1, "type": "invalid_request_error"}});
        (self.0, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

pub async fn serve(
    engine: Engine,
    model_id: String,
    batch: usize,
    host: &str,
    port: u16,
) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        model_id,
        batch,
    });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/ocr", post(ocr))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(256 << 20))
        .with_state(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{addr}");
    eprintln!("teleocr server on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn models(State(s): State<Shared>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": s.model_id, "object": "model", "owned_by": "teleocr-rs"}]
    }))
}

/// Decodes base64 or a `data:` URI.
fn decode_image_b64(s: &str) -> Result<Vec<u8>, ApiError> {
    let b64 = match s.strip_prefix("data:") {
        Some(rest) => rest
            .split_once(',')
            .map(|(_, d)| d)
            .ok_or_else(|| bad("malformed data URI"))?,
        None => s,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| bad(format!("invalid base64 image: {e}")))
}

fn load_image(bytes: &[u8]) -> Result<image::RgbImage, ApiError> {
    Ok(image::load_from_memory(bytes)
        .map_err(|e| bad(format!("unreadable image: {e}")))?
        .to_rgb8())
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OcrRequest {
    /// base64 or data URI.
    image: Option<String>,
    /// parse (default) | text | table | formula | code | layout | layout_seg | figure | seal
    task: Option<String>,
    /// Free prompt; overrides `task` for a single-prompt run.
    prompt: Option<String>,
    mode: Option<LayoutMode>,
    paratext: Option<bool>,
    max_tokens: Option<usize>,
    /// Resize the image to NxN first (single-prompt tasks).
    resize: Option<u32>,
    /// PDF pages, 1-based ("1-3,5"); default all.
    pages: Option<String>,
    /// PDF render resolution (default 200).
    dpi: Option<f32>,
}

async fn ocr(
    State(s): State<Shared>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, ApiError> {
    let ctype = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let (req, bytes) = if ctype.starts_with("multipart/form-data") {
        parse_multipart(ctype, body).await?
    } else {
        let req: OcrRequest =
            serde_json::from_slice(&body).map_err(|e| bad(format!("invalid JSON: {e}")))?;
        let bytes = decode_image_b64(req.image.as_deref().ok_or_else(|| bad("missing 'image'"))?)?;
        (req, bytes)
    };
    let task = req.task.clone().unwrap_or_else(|| "parse".into());
    let t0 = Instant::now();
    if bytes.starts_with(b"%PDF") {
        return parse_pdf(s, req, bytes, t0).await;
    }
    let img = load_image(&bytes)?;

    let state = s.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<Value> {
        let engine = state.engine.blocking_lock();
        if task == "parse" && req.prompt.is_none() {
            let opts = ParseOptions {
                mode: req.mode.unwrap_or(LayoutMode::Detection),
                paratext: req.paratext.unwrap_or(false),
                max_new_tokens: req.max_tokens.unwrap_or(4096),
                batch: state.batch,
            };
            let page = engine.parse_page(&img, &opts)?;
            return Ok(serde_json::to_value(page)?);
        }
        let prompt = match &req.prompt {
            Some(p) => p.clone(),
            None => single_task_prompt(&task)
                .ok_or_else(|| anyhow!("unknown task '{task}'"))?
                .to_string(),
        };
        let mut img = img;
        let resize = req
            .resize
            .or(task.starts_with("layout").then_some(pipeline::LAYOUT_SIZE));
        if let Some(n) = resize {
            img = teleocr::image::resize_bicubic(&img, n as usize, n as usize);
        }
        let mut opts = GenerateOptions::pipeline(pipeline_kind(&task));
        opts.max_new_tokens = req.max_tokens.unwrap_or(4096);
        let out = engine.generate(&img, &prompt, &opts, None)?;
        let content = postprocess_single(&task, &out.text);
        Ok(json!({
            "task": task,
            "raw": out.text,
            "content": content,
            "truncated": out.truncated,
            "timings": out.timings,
        }))
    })
    .await
    .map_err(|e| anyhow!("worker: {e}"))?
    .map_err(|e| {
        let msg = format!("{e:#}");
        if msg.starts_with("unknown task") {
            bad(msg)
        } else {
            ApiError(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    })?;
    let mut result = result;
    result["elapsed_ms"] = json!(t0.elapsed().as_secs_f64() * 1e3);
    Ok(Json(result))
}

/// Every selected page of a PDF through the document pipeline.
async fn parse_pdf(
    s: Shared,
    req: OcrRequest,
    bytes: Vec<u8>,
    t0: Instant,
) -> Result<Json<Value>, ApiError> {
    if req.task.as_deref().is_some_and(|t| t != "parse") || req.prompt.is_some() {
        return Err(bad("PDF input supports task=parse only"));
    }
    let images = crate::load_document(&bytes, req.pages.as_deref(), req.dpi.unwrap_or(200.0))
        .map_err(|e| bad(format!("{e:#}")))?;
    let opts = ParseOptions {
        mode: req.mode.unwrap_or(LayoutMode::Detection),
        paratext: req.paratext.unwrap_or(false),
        max_new_tokens: req.max_tokens.unwrap_or(4096),
        batch: s.batch,
    };
    let pages = tokio::task::spawn_blocking(move || -> Result<Vec<teleocr::Page>> {
        let engine = s.engine.blocking_lock();
        engine.parse_pages(&images, &opts)
    })
    .await
    .map_err(|e| anyhow!("worker: {e}"))??;
    let markdown = pages
        .iter()
        .map(|p| p.markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(Json(json!({
        "markdown": markdown,
        "pages": pages,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3,
    })))
}

async fn parse_multipart(
    ctype: &str,
    body: axum::body::Bytes,
) -> Result<(OcrRequest, Vec<u8>), ApiError> {
    let boundary = multer::parse_boundary(ctype).map_err(|e| bad(format!("multipart: {e}")))?;
    let stream = futures_util::stream::once(async move { Ok::<_, std::io::Error>(body) });
    let mut mp = multer::Multipart::new(stream, boundary);
    let mut req = OcrRequest::default();
    let mut bytes = None;
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| bad(format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "image" {
            bytes = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| bad(e.to_string()))?
                    .to_vec(),
            );
            continue;
        }
        let v = field.text().await.map_err(|e| bad(e.to_string()))?;
        match name.as_str() {
            "task" => req.task = Some(v),
            "prompt" => req.prompt = Some(v),
            "mode" => {
                req.mode = Some(
                    serde_json::from_value(json!(v))
                        .map_err(|_| bad("mode: detection|segmentation"))?,
                )
            }
            "paratext" => req.paratext = Some(v == "true" || v == "1"),
            "max_tokens" => req.max_tokens = v.parse().ok(),
            "resize" => req.resize = v.parse().ok(),
            "pages" => req.pages = Some(v),
            "dpi" => req.dpi = v.parse().ok(),
            _ => {}
        }
    }
    Ok((req, bytes.ok_or_else(|| bad("missing 'file' field"))?))
}

/// Prompts of the single-shot tasks (the pipeline's prompts, with the
/// leading newline the official client uses).
fn single_task_prompt(task: &str) -> Option<&'static str> {
    Some(match task {
        "text" => pipeline::block_prompt("text"),
        "table" => pipeline::block_prompt("table"),
        "formula" | "equation" => pipeline::block_prompt("equation"),
        "code" => pipeline::block_prompt("code"),
        "seal" => pipeline::block_prompt("seal"),
        "figure" | "char" => pipeline::block_prompt("char"),
        "layout" => "\nAnalyze the image layout.",
        "layout_seg" => "\nMulti-point Layout Segmentation Analysis.",
        _ => return None,
    })
}

fn pipeline_kind(task: &str) -> &str {
    match task {
        "formula" => "equation",
        "figure" => "char",
        t => t,
    }
}

fn postprocess_single(task: &str, text: &str) -> Value {
    match task {
        "table" | "figure" | "char" => json!(pipeline::post_table(text)),
        "formula" | "equation" => json!(pipeline::post_equation(text)),
        "text" => json!(pipeline::post_text(text)),
        "code" => {
            let (lang, code) = pipeline::split_code(text);
            json!({"lang": lang, "code": code})
        }
        "layout" | "layout_seg" => json!(pipeline::parse_layout(text)),
        _ => json!(text),
    }
}

// ------------------------------------------------------------ chat completions

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: Value,
}

async fn chat(State(s): State<Shared>, Json(req): Json<ChatRequest>) -> Result<Response, ApiError> {
    if req.stream {
        return Err(bad("stream=true is not supported; use stream=false"));
    }
    // Last user message: one image + text.
    let msg = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .ok_or_else(|| bad("no user message"))?;
    let mut text = String::new();
    let mut image_bytes = None;
    match &msg.content {
        Value::String(s) => text.push_str(s),
        Value::Array(parts) => {
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        text.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some("image_url") => {
                        let url = p
                            .get("image_url")
                            .and_then(|u| u.get("url").or(Some(u)))
                            .and_then(|u| u.as_str())
                            .ok_or_else(|| bad("image_url.url missing"))?;
                        if image_bytes.is_some() {
                            return Err(bad("only one image per request"));
                        }
                        if !url.starts_with("data:") {
                            return Err(bad("image_url must be a data: URI"));
                        }
                        image_bytes = Some(decode_image_b64(url)?);
                    }
                    _ => {}
                }
            }
        }
        _ => return Err(bad("unsupported message content")),
    }
    let img = load_image(&image_bytes.ok_or_else(|| bad("an image is required"))?)?;
    let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(4096);
    let task = text.trim().to_lowercase();
    let t0 = Instant::now();

    let state = s.clone();
    let (content, out) =
        tokio::task::spawn_blocking(move || -> Result<(String, teleocr::Output)> {
            let engine = state.engine.blocking_lock();
            // A bare task name (or empty text) maps to the task prompt.
            let (prompt, kind) = if task.is_empty() {
                (
                    single_task_prompt("text").unwrap().to_string(),
                    "text".to_string(),
                )
            } else if let Some(p) = single_task_prompt(&task) {
                (p.to_string(), task.clone())
            } else {
                (text.clone(), "default".to_string())
            };
            let mut img = img;
            if kind.starts_with("layout") {
                let n = pipeline::LAYOUT_SIZE as usize;
                img = teleocr::image::resize_bicubic(&img, n, n);
            }
            let mut opts = GenerateOptions::pipeline(pipeline_kind(&kind));
            opts.max_new_tokens = max_tokens;
            let out = engine.generate(&img, &prompt, &opts, None)?;
            Ok((out.text.clone(), out))
        })
        .await
        .map_err(|e| anyhow!("worker: {e}"))??;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let t = &out.timings;
    Ok(Json(json!({
        "id": format!("chatcmpl-{created}{}", t0.elapsed().as_micros() % 1_000_000),
        "object": "chat.completion",
        "created": created,
        "model": s.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": if out.truncated { "length" } else { "stop" },
        }],
        "usage": {
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.generated_tokens,
            "total_tokens": t.prompt_tokens + t.generated_tokens,
        },
        "timings": t,
    }))
    .into_response())
}
