# AGENTS.md

Guidance for AI coding agents working in this repo.

**What this is:** a C# (.NET 10 + TorchSharp) reader and merger for PEFT LoRA adapters, verified
against Python PEFT. `W' = W0 + (alpha/r) * B @ A`.

**Map**
- `src/LoraAnatomy/SafeTensors.cs` -- safetensors reader (F32, F64, F16, BF16)
- `src/LoraAnatomy/LoraMerge.cs` -- scaling, merge, checkpoint merge, inert-adapter check (pure, unit-tested)
- `src/LoraAnatomy/SafeTensorsWriter.cs`, `Cli.cs` -- writer and the `check` / `merge` subcommands
- `src/LoraAnatomy/Program.cs` -- demo: inspect + merge + compare to PEFT ground truth
- `reference/` -- Python ground truth (`uv run ...`); `tests/LoraAnatomy.Tests` -- unit tests
- `GOTCHAS.md` -- read before touching merge logic

**Commands:** `dotnet test LoraAnatomy.slnx` (fast, offline). `cd src/LoraAnatomy && dotnet run` needs `reference/out` from `make_reference.py`.

**Invariants:** scaling is `alpha/r` (or `alpha/sqrt(r)` if `use_rslora`); `use_rslora` may be absent; adapter keys carry the `base_model.model.` prefix; a verified merge must also change behaviour, not just match bytes.

**Done:** `check` and `merge` CLI, safetensors writer, F16/BF16 decode.

**Good next steps:** sharded base checkpoints (`model.safetensors.index.json`); writing bf16/f16 instead of always f32; int8/other dtypes; DoRA; multi-adapter composition; a NuGet package / `dotnet tool`.
