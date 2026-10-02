//! TeleOCR (~1.2B document-parsing VLM) in Rust on Candle.
//!
//! Architecture: Qwen2.5-VL vision tower (window attention, 2x2 merger) feeding
//! a Qwen3-style text decoder with multimodal RoPE. Weights load from a
//! Hugging Face directory or a single GGUF file (see [`gguf`]).

pub mod config;
pub mod gguf;
pub mod image;
pub mod model;
pub mod otsl;
pub mod pipeline;
pub mod text;
pub mod vision;
pub mod weights;

pub use model::{Engine, GenerateOptions, LoadOptions, Output, task_prompt};
pub use pipeline::{LayoutMode, Page, ParseOptions};
