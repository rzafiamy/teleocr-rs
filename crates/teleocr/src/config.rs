//! Model hyper-parameters, read from the Hugging Face `config.json`
//! (and stored verbatim in the GGUF file).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub window_size: usize,
    pub fullatt_block_indexes: Vec<usize>,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
}

fn default_in_channels() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_config: VisionConfig,
}

impl Config {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let mut cfg: Config = serde_json::from_str(s)?;
        // Some exports keep the text hyper-parameters only in `text_config`.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s)
            && let Some(tie) = v
                .get("text_config")
                .and_then(|t| t.get("tie_word_embeddings"))
                .and_then(|t| t.as_bool())
        {
            cfg.tie_word_embeddings |= tie;
        }
        let half: usize = cfg.rope_scaling.mrope_section.iter().sum();
        anyhow::ensure!(
            half * 2 == cfg.head_dim,
            "mrope_section {:?} does not cover head_dim {}",
            cfg.rope_scaling.mrope_section,
            cfg.head_dim
        );
        Ok(cfg)
    }
}
