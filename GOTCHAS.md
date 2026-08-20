# GOTCHAS

Raw log. Appended while bleeding, not cleaned up afterwards.
Tagged `[HIT]` when it actually bit, `[TRAP]` when it was anticipated and dodged
but would have bitten someone — worth keeping either way.

---

### [TRAP] PEFT zero-initialises `lora_B`, so a fresh adapter merges to nothing

`get_peft_model()` gives you `lora_B = 0`. Since `ΔW = BA`, a freshly created
adapter has `ΔW = 0` exactly. Merge it and *nothing happens* — `W' == W0`.

This is correct and deliberate (training must start from the unmodified model),
but it means a naive "does my merge work?" test passes vacuously: you compare
zero against zero and declare victory.

Fix in `reference/make_reference.py`: perturb `lora_B` with random values before
merging, then assert `delta.abs().max() != 0` before writing ground truth.

**Never verify a merge against an untrained adapter.**

---

### [HIT] The scaling factor is `alpha/r`, and forgetting it is nearly invisible

The merge is **not** `W' = W0 + BA`. It is:

```
W' = W0 + (alpha / r) * BA
```

Numbers from this repo (`r=8`, `alpha=16`, so `scaling = 2`):

```
magnitude of the real delta        : 3.258E-002
max abs error  W0 + BA             : 1.629E-002   <-- forgot alpha/r
max abs error  W0 + (alpha/r) * BA : 0.000E+000
```

The dangerous part is the ratio. The error is **the same order of magnitude as
the delta itself** — not 1000x too big, not obviously broken. A model merged this
way loads fine, generates fluent text, and is simply *wrong by half*. There is no
crash to lead you to it.

Section 6 of the study guide writes the equation as `W = W0 + BA`, which is the
conceptual form. The implementation form has the scaling in it.

Also note: with `use_rslora: true` the scaling becomes `alpha/sqrt(r)` instead.
Check `adapter_config.json` rather than assuming.

---

### [HIT] A popular published adapter on the Hub is inert — all 48 `lora_B` are zero

The zero-init trap above is not hypothetical. `ybelkada/opt-350m-lora`
(~118k downloads, third-most-downloaded PEFT adapter at time of writing) ships
with **every one of its 48 `lora_B` tensors exactly zero**.

```
lora_A tensors: 48, all-zero:  0
lora_B tensors: 48, all-zero: 48
```

Since `ΔW = BA`, `ΔW = 0` for every module. Loading it, merging it, and serving
it is a guaranteed no-op — you get the base model back, silently. It is presumably
published as a structural example rather than a trained adapter, but nothing in
the repo or config says so.

`reference/scan_adapters.py` checks a list of adapters for this. Of 5 that could
be scanned (3 others ship `.bin` pickles rather than safetensors), 1 was inert.
Not an epidemic — but the inert one is the one a newcomer is most likely to grab
as a first example.

**Check `max(|B|) > 0` before you trust any merge, yours or anyone else's.**

---

### [HIT] `use_rslora` is often absent from the config entirely

Older adapters have no `use_rslora` key. Reading it blindly throws; the correct
behaviour is to default to `false`. When it *is* true the divisor changes:

```
standard : scaling = alpha / r
rsLoRA   : scaling = alpha / sqrt(r)
```

---

### [HIT] When `alpha == r`, the scaling bug is completely invisible

`lucylq/qwen3_06B_lora_math` uses `r=16, alpha=16`, so `scaling == 1.0` and
`W0 + BA` is *identical* to `W0 + (alpha/r)·BA`. A wrong implementation passes
every test on this adapter.

That is how the bug survives: it is dormant on any adapter where `alpha == r`,
and wakes up the moment someone loads one where `alpha = 2r` (a very common
default). Your test adapter can certify a broken merge as correct.

---

### [HIT] LoRA targets are not all square

The toy model only had `hidden x hidden` attention projections, which hid this.
On a real model:

```
q_proj      1024 x 1024
down_proj   1024 x 3072    <-- in_features != hidden_size
```

A probe vector sized to `hidden_size` blows up on `down_proj`:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1024 and 3072x1024)
```

Size activations from `weight.shape[1]`, never from `config.hidden_size`. This
also means shape-checking is a weaker guard against transposition mistakes than
it looks on a square layer.

---

### [HIT] The tensor names have `model` in them three times

```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
```

`base_model.` from the PEFT wrapper, `model.` from `LlamaForCausalLM`, `model.`
again from the inner `LlamaModel`. Not a typo, and string-matching on
`"model."` will do something you did not intend.

---

### [HIT] It is `B @ A`, not `A @ B` — the shapes are the only thing stopping you

```
A: [8, 64]   (r x k)
B: [64, 8]   (d x r)
B @ A -> [64, 64]  = W0's shape
```

Matches guide §7. `A @ B` would give `[8,8]` and fail loudly — but only because
this is a square projection. On a non-square weight the wrong order can still
produce a plausible-looking shape.

---

### [NOTE] TorchSharp on Apple Silicon needed no fighting

Expected native-library pain, got none. `dotnet add package TorchSharp-cpu`
(0.107.0), `dotnet run`, matmul works, `cuda.is_available() == False` as expected.
Worth recording precisely because the internet led me to expect otherwise.

---

### [HIT] A matching merge is not proof the merge did anything

Every module reported `0.000E+000` error against PEFT — and that number is
equally consistent with "the merge is correct" and "the adapter is a no-op and I
compared `W0` to `W0`". The first gotcha in this file is exactly that failure
mode.

So the forward pass is not decoration. Push a fixed activation `x` through `W0`
and through `W'` and show the output moved:

```
|y_base| max            : 2.721E-001
|y_merged - y_base| max  : 2.054E-001   (75.5% of signal)
cosine(y_base, y_merged) : 0.879816
```

Two assertions, not one: **the merged activation matches PEFT's**, and **it
differs from the base**. Either alone can pass while the implementation is
broken.

Caveat on that 75.5%: `lora_B` here was filled with `N(0, 0.02)` against a
randomly-initialised base, so the perturbation is far larger relative to the
signal than a trained adapter would be. Treat it as "the wiring works", not as a
realistic effect size.

The real adapter confirms it. Same measurement on `lucylq/qwen3_06B_lora_math`:

```
|y_merged - y_base| max  : 7.979E-002   (1.5% of signal)
cosine(y_base, y_merged) : 0.999897
```

**1.5%, not 75%.** A trained LoRA is a very small nudge to any individual
projection — which is a useful thing to have felt, because it means an
`assert outputs_changed_a_lot` style check would fail on a perfectly good
adapter, and a merge that is wrong by a few percent looks exactly like a merge
that is right.

---

### [NOTE] The parameter reduction here looks unimpressive on purpose

This repo reports `4x smaller`, not the `256x` from guide §7. The toy model uses
`d = k = 64`; the guide's example uses `4096`. The reduction ratio scales with
`d*k / (r*(d+k))`, so it only becomes dramatic at real model widths. Do not quote
a reduction factor measured on a toy.
