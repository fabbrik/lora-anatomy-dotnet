using static TorchSharp.torch;

namespace LoraAnatomy;

/// <summary>The LoRA merge equation, isolated so it can be tested without any model files.</summary>
public static class LoraMerge
{
    /// <summary>
    /// The factor applied to <c>B @ A</c>. Standard LoRA uses <c>alpha / r</c>; rsLoRA
    /// (<c>use_rslora: true</c>) uses <c>alpha / sqrt(r)</c>.
    /// </summary>
    public static double Scaling(int r, int alpha, bool useRsLora = false)
    {
        if (r <= 0) throw new ArgumentOutOfRangeException(nameof(r), "rank must be positive");
        return useRsLora ? alpha / Math.Sqrt(r) : (double)alpha / r;
    }

    /// <summary><c>W' = W0 + scaling * B @ A</c>. Shapes: W0 (d,k), A (r,k), B (d,r).</summary>
    public static Tensor Merge(Tensor w0, Tensor a, Tensor b, double scaling)
    {
        if (a.shape[0] != b.shape[1])
            throw new ArgumentException($"rank mismatch: A is [{string.Join(",", a.shape)}], B is [{string.Join(",", b.shape)}]");
        if (w0.shape[0] != b.shape[0] || w0.shape[1] != a.shape[1])
            throw new ArgumentException($"W0 [{string.Join(",", w0.shape)}] does not fit B@A [{b.shape[0]},{a.shape[1]}]");
        return w0 + scaling * b.matmul(a);
    }

    /// <summary>
    /// True when every element of <c>lora_B</c> is exactly zero, so merging is a guaranteed
    /// no-op (<c>BA = 0</c>). PEFT initialises B to zero, and some published adapters never left it.
    /// </summary>
    public static bool IsInert(Tensor loraB) => loraB.abs().max().item<float>() == 0f;

    private const string PeftPrefix = "base_model.model.";

    /// <summary>
    /// The base-checkpoint key an adapter module patches: the PEFT wrapper prefix is dropped and
    /// <c>.weight</c> appended, so <c>base_model.model.model.layers.0.self_attn.q_proj</c> patches
    /// <c>model.layers.0.self_attn.q_proj.weight</c>.
    /// </summary>
    public static string BaseKeyFor(string module) =>
        (module.StartsWith(PeftPrefix, StringComparison.Ordinal) ? module[PeftPrefix.Length..] : module) + ".weight";

    /// <summary>
    /// Merge every module in an adapter into a base checkpoint and return the whole patched
    /// checkpoint. Tensors the adapter does not touch pass through unchanged. A module the
    /// adapter targets but the base lacks is an error: silently skipping it would produce a
    /// "merged" model that is partly just the base.
    /// </summary>
    public static (Dictionary<string, Tensor> Merged, IReadOnlyList<string> PatchedKeys) MergeCheckpoint(
        IReadOnlyDictionary<string, Tensor> baseWeights, IReadOnlyDictionary<string, Tensor> adapter, double scaling)
    {
        var merged = new Dictionary<string, Tensor>(baseWeights);
        var patched = new List<string>();
        var modules = adapter.Keys
            .Where(k => k.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            .Select(k => k[..^".lora_A.weight".Length])
            .OrderBy(k => k, StringComparer.Ordinal);

        foreach (string module in modules)
        {
            string key = BaseKeyFor(module);
            if (!baseWeights.TryGetValue(key, out Tensor? w0))
                throw new KeyNotFoundException($"adapter module '{module}' targets '{key}', which the base checkpoint does not contain.");
            if (!adapter.TryGetValue($"{module}.lora_B.weight", out Tensor? b))
                throw new KeyNotFoundException($"adapter module '{module}' has lora_A but no lora_B.");

            merged[key] = Merge(w0, adapter[$"{module}.lora_A.weight"], b, scaling);
            patched.Add(key);
        }

        return (merged, patched);
    }
}
