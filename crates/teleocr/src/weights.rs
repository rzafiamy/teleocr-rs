//! Weight store shared by the safetensors and GGUF loaders, and the linear
//! layer that runs either dense or quantized (`QMatMul`).

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Weights {
    map: HashMap<String, Arc<QTensor>>,
    pub device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, Arc<QTensor>>, device: Device) -> Self {
        Self { map, device }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<&Arc<QTensor>> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor '{name}'"))
    }

    /// Dense tensor in `dtype` on the model device.
    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let q = self.raw(name)?;
        let t = q.dequantize(&q.device())?.to_dtype(dtype)?;
        Ok(t.to_device(&self.device)?)
    }

    /// The tensor as a `QMatMul` when it is block-quantized.
    pub fn qmatmul(&self, name: &str) -> Result<Option<QMatMul>> {
        let q = self.raw(name)?;
        Ok(match q.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => None,
            _ => Some(QMatMul::from_arc(q.clone())?),
        })
    }

    pub fn linear(&self, prefix: &str, bias: bool, dtype: DType) -> Result<Linear> {
        let name = format!("{prefix}.weight");
        let q = self.raw(&name)?;
        let bias = if bias {
            Some(self.get(&format!("{prefix}.bias"), dtype)?)
        } else {
            None
        };
        let inner = match q.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                // Convert on the CPU, then move: one device copy only.
                let t = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
                LinearInner::Dense(t.to_device(&self.device)?)
            }
            _ => LinearInner::Quant(QMatMul::from_arc(q.clone())?),
        };
        Ok(Linear { inner, bias })
    }
}

#[derive(Debug, Clone)]
enum LinearInner {
    /// `[out, in]` weight in the layer's compute dtype.
    Dense(Tensor),
    /// Block-quantized weight; takes and returns F32 activations.
    Quant(QMatMul),
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
    bias: Option<Tensor>,
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let ys = match &self.inner {
            LinearInner::Dense(w) => {
                // Flatten to 2-D: candle 0.11 matmul against a stride-0
                // (broadcast_left) batch returns wrong values.
                let dims = xs.dims().to_vec();
                let k = dims[dims.len() - 1];
                let x2 = xs.to_dtype(w.dtype())?.reshape(((), k))?;
                let y = x2.matmul(&w.t()?)?;
                let mut out = dims;
                *out.last_mut().unwrap() = w.dim(0)?;
                y.reshape(out)?
            }
            LinearInner::Quant(q) => {
                let in_dtype = xs.dtype();
                q.forward(&xs.to_dtype(DType::F32)?)?.to_dtype(in_dtype)?
            }
        };
        match &self.bias {
            Some(b) => ys.broadcast_add(&b.to_dtype(ys.dtype())?),
            None => Ok(ys),
        }
    }
}

/// RMSNorm computed in F32, output cast back to the input dtype
/// (matches `Qwen2RMSNorm` / `Qwen3RMSNorm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(w: &Weights, name: &str, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{name}.weight"), dtype)?,
            eps,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = xs.dtype();
        let x = candle_nn::ops::rms_norm(
            &xs.to_dtype(DType::F32)?.contiguous()?,
            &self.weight.to_dtype(DType::F32)?,
            self.eps as f32,
        )?;
        x.to_dtype(dtype)
    }
}
