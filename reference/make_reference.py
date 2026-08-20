"""Build a real PEFT LoRA adapter and dump ground-truth tensors for the C# side.

Deliberately offline: the base model is constructed from a config with random
weights, so no download is needed. The adapter is still produced by real PEFT,
so all of PEFT's conventions (tensor naming, alpha/r scaling, merge behaviour)
are genuine -- which is the whole point of the comparison.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

TARGETS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
]
RANK = 8
ALPHA = 16

torch.manual_seed(0)

# --- 1. base model -----------------------------------------------------------
cfg = LlamaConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)
base = LlamaForCausalLM(cfg).eval()

w0 = {t: base.get_submodule(t).weight.detach().clone() for t in TARGETS}
for t, w in w0.items():
    print(f"W0 {t}.weight shape={tuple(w.shape)}")

# A fixed input vector, so the C# side can check that the merged projection
# actually moves an activation -- numerical equality is not behavioural proof.
x = torch.randn(1, cfg.hidden_size)

# --- 2. wrap with a real LoRA adapter ---------------------------------------
lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(base, lora_cfg)

# GOTCHA: PEFT zero-initialises lora_B, so delta_W == 0 on a freshly created
# adapter. Merging an untrained adapter is a no-op and would make the whole
# comparison vacuously pass. Perturb B so there is a real delta to verify.
with torch.no_grad():
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            param.normal_(mean=0.0, std=0.02)

peft_model.save_pretrained(str(OUT / "adapter"))

# --- 3. merge and capture ground truth --------------------------------------
merged_model = peft_model.merge_and_unload()

tensors = {"x": x.contiguous()}
meta_targets = {}

for t in TARGETS:
    w_merged = merged_model.get_submodule(t).weight.detach().clone()
    delta = w_merged - w0[t]
    print(f"delta {t}: max_abs={delta.abs().max():.6e}")
    if delta.abs().max() == 0:
        raise SystemExit(f"delta is zero for {t} -- nothing to verify")

    tensors[f"{t}.w0"] = w0[t].contiguous()
    tensors[f"{t}.w_merged"] = w_merged.contiguous()
    meta_targets[t] = float(delta.abs().max())

# Reference activations for the forward-pass check on the first target.
first = TARGETS[0]
tensors["y_base"] = (x @ w0[first].T).contiguous()
tensors["y_merged"] = (x @ tensors[f"{first}.w_merged"].T).contiguous()

save_file(tensors, str(OUT / "ground_truth.safetensors"))

(OUT / "meta.json").write_text(
    json.dumps(
        {
            "targets": TARGETS,
            "forward_target": first,
            "r": RANK,
            "lora_alpha": ALPHA,
            "scaling": ALPHA / RANK,
            "delta_max_abs": meta_targets,
        },
        indent=2,
    )
)

print(f"\nwrote {OUT}/ground_truth.safetensors")
print(f"wrote {OUT}/adapter/  (adapter_model.safetensors + adapter_config.json)")
