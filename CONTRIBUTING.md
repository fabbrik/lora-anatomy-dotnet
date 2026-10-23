# Contributing

Small, sharp contributions are welcome: another dtype, another adapter format quirk, a
published adapter that behaves oddly, a correction to GOTCHAS.md.

```bash
dotnet test LoraAnatomy.slnx          # unit tests, no model files needed
cd reference && uv run make_reference.py && cd ../src/LoraAnatomy && dotnet run   # PEFT cross-check
```

Ground rules:

- **A merge claim needs ground truth.** If you change merge behaviour, the synthetic run must
  still report `MATCH` against Python PEFT, and a unit test must pin the new behaviour.
- **Never verify against an untrained adapter** (`lora_B == 0`). It passes vacuously. See GOTCHAS.md.
- **Fail loudly.** Unknown dtype, truncated file, shape mismatch: throw with a message that says what to do.
- Log anything that bit you in GOTCHAS.md, tagged `[HIT]` or `[TRAP]`.
