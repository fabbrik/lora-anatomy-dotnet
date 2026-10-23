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
}
