#!/usr/bin/env python3
"""TeleOCR Python reference: dumps intermediates for parity checks of the Rust port.

usage: ref.py <hf-dir> <image> <prompt-key> <out.safetensors> [--resize 1036] [--max-new 512]
"""
import argparse, json, sys, time

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor

PROMPTS = {
    "text": "Please output the text content from the image.",
    "table": "This is the image of a table. Please output the table in OTSL format.",
    "formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "code": "The image contains a code snippet, please output the parsing result.",
    "layout": "Analyze the image layout.",
    "layout_seg": "\nMulti-point Layout Segmentation Analysis.",
    "figure": "This is a scientific figure. Please extract the table implied by this figure.",
}

ap = argparse.ArgumentParser()
ap.add_argument("hf")
ap.add_argument("image")
ap.add_argument("prompt")
ap.add_argument("out")
ap.add_argument("--resize", type=int, default=0)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads)

proc = AutoProcessor.from_pretrained(a.hf, trust_remote_code=True, use_fast=True)
model = AutoModel.from_pretrained(a.hf, trust_remote_code=True, torch_dtype=torch.float32).eval()

img = Image.open(a.image).convert("RGB")
if a.resize:
    img = img.resize((a.resize, a.resize), Image.Resampling.BICUBIC)
prompt = PROMPTS.get(a.prompt, a.prompt)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
]
chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[chat], images=[img], padding=True, return_tensors="pt")

dump = {
    "input_ids": inputs.input_ids[0].to(torch.int64),
    "pixel_values": inputs.pixel_values.float(),
    "image_grid_thw": inputs.image_grid_thw.to(torch.int64),
}
with torch.no_grad():
    t0 = time.time()
    vis = model.model.visual(inputs.pixel_values.float(), grid_thw=inputs.image_grid_thw)
    dump["vision_out"] = vis.float()
    print(f"vision {tuple(vis.shape)} {time.time()-t0:.2f}s", file=sys.stderr)
    pos, delta = model.model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, None, None, inputs.attention_mask)
    dump["position_ids"] = pos[:, 0, :].to(torch.int64)
    out = model(**inputs)
    dump["prefill_logits_last"] = out.logits[0, -1].float()
    t0 = time.time()
    gen = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False, use_cache=True)
    print(f"generate {time.time()-t0:.2f}s", file=sys.stderr)
new = gen[0, inputs.input_ids.shape[1]:]
dump["generated"] = new.to(torch.int64).contiguous()
text = proc.batch_decode([new.tolist()], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
save_file({k: v.contiguous() for k, v in dump.items()}, a.out, metadata={"text": text, "prompt": prompt})
print(json.dumps({"tokens": len(new), "text": text}, ensure_ascii=False))
