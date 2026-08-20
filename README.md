# lora-anatomy-dotnet

What is actually inside a LoRA adapter file, and how to merge one into base
weights **in C#** — verified against Python PEFT's own merge.

No wrapper, no bindings to a Python process. A safetensors reader, TorchSharp
tensors, and the equation from the paper:

```
W' = W0 + (alpha / r) * B @ A
```

## Why

The .NET AI ecosystem can *run* models ([LLamaSharp](https://github.com/SciSharp/LLamaSharp),
[dotLLM](https://github.com/kkokosa/dotLLM), ONNX Runtime) but there is
essentially nothing published about *adapting* them. ML.NET's deep-learning
surface stops at NAS-BERT text classification with a 512-token cap — no LoRA,
no adapter loading. This repo starts filling that gap at the smallest possible
unit: read the file, understand it, merge it, prove the merge is right.

## Run it

Two datasets. The synthetic one is offline and instant; the real one downloads a
published adapter and its base model.

```bash
cd reference

# A. synthetic -- real PEFT conventions, toy model built from config, no downloads
uv run make_reference.py
cd ../src/LoraAnatomy && dotnet run

# B. real -- lucylq/qwen3_06B_lora_math on unsloth/Qwen3-0.6B (~1.2GB, cached)
cd ../../reference && uv run make_real_reference.py
cd ../src/LoraAnatomy && dotnet run -- real

# C. are published adapters actually trained?
cd ../../reference && uv run scan_adapters.py
```

## Output

```
=== adapter_config.json ===
  r            = 8
  lora_alpha   = 16
  scaling      = alpha/r = 2

=== adapter tensors ===
  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight   [8,64]
  base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight   [64,8]
  ...
  8 tensors, 4,096 trainable parameters total

=== merge + verification against Python PEFT ===
  module                          delta        W0+BA   W0+(a/r)BA
  L0.q_proj                  3.364E-002   1.682E-002   0.000E+000
  L0.v_proj                  4.215E-002   2.108E-002   0.000E+000
  L1.q_proj                  3.021E-002   1.510E-002   0.000E+000
  L1.v_proj                  3.176E-002   1.588E-002   0.000E+000

=== forward pass through L0.q_proj ===
  C# merged activation vs PEFT's : 0.000E+000   (max abs error)
  |y_merged - y_base| max        : 2.054E-001   (75.5% of signal)
  cosine(y_base, y_merged)       : 0.879816

MATCH: the C# merge reproduces PEFT's merge on every targeted module.
```

The `W0+BA` column is the whole point. Forgetting the `alpha/r` scaling produces
an error **the same order of magnitude as the effect you were trying to apply** —
the model loads, generates fluent text, and is wrong by half, with nothing to
alert you.

The forward pass is the second half of the proof: a merge that matches PEFT
exactly is *also* consistent with the adapter being a no-op. So the repo asserts
both that the merged activation matches PEFT's **and** that it differs from the
base. See [GOTCHAS.md](GOTCHAS.md).

## One finding worth its own line

`ybelkada/opt-350m-lora` — around 118k downloads, one of the most-downloaded
PEFT adapters on the Hub — has **all 48 of its `lora_B` matrices set to exactly
zero**. Since `ΔW = BA`, merging it is a guaranteed no-op: you get the base model
back, silently, with no error. `reference/scan_adapters.py` will tell you whether
any given adapter is actually trained.

## Layout

```
reference/                  Python ground truth (torch + transformers + peft)
  make_reference.py           toy model built from config, offline
  make_real_reference.py      published Hub adapter + its real base model
  scan_adapters.py            checks published adapters for all-zero lora_B
src/LoraAnatomy/            .NET 10 + TorchSharp
  SafeTensors.cs              minimal safetensors reader
  Program.cs                  inspect, reconstruct delta-W, merge, verify
GOTCHAS.md                  what bit, with the numbers
```

## Stack

- **.NET 10** + **TorchSharp-cpu** 0.107.0 (works on Apple Silicon with no setup)
- **Python 3.11** + torch / transformers / peft, managed with `uv`

TensorFlow.NET was considered and rejected: no active maintainers, pinned to
TF 2.10 / CUDA 11, and mismatched with a PEFT ecosystem that is PyTorch-native
end to end.

## Status

Rung 1 of a ladder. Next: a multi-tenant adapter hot-swap server on ASP.NET —
one base model, per-request adapter selection.
