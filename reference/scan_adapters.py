"""Scan published LoRA adapters for all-zero lora_B matrices.

Since delta_W = B @ A, an adapter whose B is entirely zero has delta_W == 0 and
merging it is a guaranteed no-op. This scans adapters on the Hub and reports
which ones are inert.
"""

import sys

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

CANDIDATES = [
    "ybelkada/opt-350m-lora",
    "peft-internal-testing/tiny-OPTForCausalLM-lora",
    "peft-internal-testing/gpt2-lora-random",
    "peft-internal-testing/opt-125m-dummy-lora",
    "llamafactory/tiny-random-Llama-3-lora",
    "dominguesm/xlm-roberta-base-lora-language-detection",
    "lucylq/qwen3_06B_lora_math",
    "oliverwang15/FinGPT_v31_ChatGLM2_Sentiment_Instruction_LoRA_FT",
]

api = HfApi()
print(f"{'adapter':<62} {'B tensors':>9} {'all-zero':>9}  verdict")
print("-" * 100)

for repo in CANDIDATES:
    try:
        files = [f.rfilename for f in api.model_info(repo).siblings]
        name = next(
            (f for f in files if f.endswith(("adapter_model.safetensors", "adapter_model.bin"))),
            None,
        )
        if name is None:
            print(f"{repo:<62} {'-':>9} {'-':>9}  no adapter file")
            continue
        if name.endswith(".bin"):
            print(f"{repo:<62} {'-':>9} {'-':>9}  .bin (pickle) -- skipped")
            continue

        path = hf_hub_download(repo, name)
        n = z = 0
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if "lora_B" in k:
                    n += 1
                    z += int(f.get_tensor(k).abs().max() == 0)

        if n == 0:
            verdict = "no lora_B found"
        elif z == n:
            verdict = "INERT -- merging is a no-op"
        elif z:
            verdict = f"PARTIALLY inert ({z}/{n} zero)"
        else:
            verdict = "trained"
        print(f"{repo:<62} {n:>9} {z:>9}  {verdict}")

    except Exception as e:  # noqa: BLE001 - a scan should survive one bad repo
        print(f"{repo:<62} {'-':>9} {'-':>9}  error: {type(e).__name__}: {e}", file=sys.stderr)
