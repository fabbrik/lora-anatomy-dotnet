"""Same ground-truth dump, but for a REAL published adapter from the Hub.

    adapter: ybelkada/opt-350m-lora   (r=16, alpha=32, q_proj + v_proj)
    base:    facebook/opt-350m

Downloads ~700MB the first time, then caches.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from peft import PeftModel

import os

# ybelkada/opt-350m-lora is deliberately NOT the default: its lora_B matrices are
# all zero, so merging it is a no-op. See GOTCHAS.md and scan_adapters.py.
ADAPTER_ID = os.environ.get("ADAPTER_ID", "lucylq/qwen3_06B_lora_math")
BASE_ID = os.environ.get("BASE_ID", "unsloth/Qwen3-0.6B")
N_TARGETS = int(os.environ.get("N_TARGETS", "6"))

OUT = Path(__file__).parent / "out-real"
OUT.mkdir(exist_ok=True)

adapter_dir = snapshot_download(ADAPTER_ID)
print(f"adapter cached at {adapter_dir}")

cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
print(f"r={cfg['r']} alpha={cfg['lora_alpha']} targets={cfg['target_modules']}")

torch.manual_seed(0)

base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32).eval()
peft_model = PeftModel.from_pretrained(base, adapter_dir)

# Which modules did the adapter actually touch?
targets = sorted(
    n.replace("base_model.model.", "").replace(".lora_A.default.weight", "")
    for n, _ in peft_model.named_parameters()
    if n.endswith(".lora_A.default.weight")
)[:N_TARGETS]
print(f"verifying {len(targets)} of the adapted modules:")
for t in targets:
    print(f"  {t}")

# NOTE: PeftModel.from_pretrained wraps each target in a LoraLayer, so the
# original weight lives at <module>.base_layer.weight, not <module>.weight.
w0 = {
    t: peft_model.get_submodule(f"base_model.model.{t}.base_layer").weight.detach().clone()
    for t in targets
}

merged = peft_model.merge_and_unload()

tensors = {}
meta = {}
for t in targets:
    wm = merged.get_submodule(t).weight.detach().clone()
    delta = wm - w0[t]
    print(f"delta {t}: max_abs={delta.abs().max():.6e}")
    if delta.abs().max() == 0:
        raise SystemExit(f"delta zero for {t} -- adapter is a no-op, nothing to verify")
    tensors[f"{t}.w0"] = w0[t].contiguous()
    tensors[f"{t}.w_merged"] = wm.contiguous()
    meta[t] = float(delta.abs().max())

# GOTCHA: LoRA targets are not all square. Attention projections are
# hidden x hidden, but down_proj is hidden x intermediate. The probe vector has
# to match the target's in_features, not the model's hidden_size.
first = targets[0]
in_features = w0[first].shape[1]
x = torch.randn(1, in_features)
print(f"\nforward probe on {first}: in_features={in_features} (hidden_size={base.config.hidden_size})")

tensors["x"] = x.contiguous()
tensors["y_base"] = (x @ w0[first].T).contiguous()
tensors["y_merged"] = (x @ tensors[f"{first}.w_merged"].T).contiguous()

save_file(tensors, str(OUT / "ground_truth.safetensors"))

# Copy the adapter next to the ground truth so the C# side has one place to look.
for name in ("adapter_config.json", "adapter_model.safetensors"):
    (OUT / name).write_bytes((Path(adapter_dir) / name).read_bytes())

(OUT / "meta.json").write_text(
    json.dumps(
        {
            "adapter": ADAPTER_ID,
            "base": BASE_ID,
            "targets": targets,
            "forward_target": first,
            "r": cfg["r"],
            "lora_alpha": cfg["lora_alpha"],
            "scaling": cfg["lora_alpha"] / cfg["r"],
            "delta_max_abs": meta,
        },
        indent=2,
    )
)

print(f"\nwrote {OUT}/")
