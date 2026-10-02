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

pub use model::{DEFAULT_KV_BUDGET, Engine, GenerateOptions, LoadOptions, Output, task_prompt};
pub use pipeline::{LayoutMode, Page, ParseOptions};

/// Returns memory the CUDA allocator keeps cached to the driver. Candle
/// allocates from the device's default stream-ordered pool, which otherwise
/// keeps the peak of past requests reserved. No-op off CUDA.
pub fn release_cached_memory(device: &candle_core::Device) -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(d) = device {
        let stream = d.cuda_stream();
        stream.synchronize()?;
        let ctx = stream.context();
        if ctx.has_async_alloc() {
            // SAFETY: the device handle comes from a live context, and the
            // default pool lives as long as the device.
            unsafe {
                let pool = cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device())?;
                cudarc::driver::result::mem_pool::trim_to(pool, 0)?;
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}
