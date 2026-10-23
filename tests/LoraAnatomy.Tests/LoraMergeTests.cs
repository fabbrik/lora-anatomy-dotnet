using LoraAnatomy;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LoraAnatomy.Tests;

public class LoraMergeTests
{
    [Theory]
    [InlineData(8, 16, false, 2.0)]
    [InlineData(4, 4, false, 1.0)]
    [InlineData(16, 32, true, 8.0)] // rsLoRA: alpha / sqrt(r)
    public void Scaling_follows_the_adapter_convention(int r, int alpha, bool rs, double expected)
        => Assert.Equal(expected, LoraMerge.Scaling(r, alpha, rs), 12);

    [Fact]
    public void Merge_matches_the_equation_by_hand()
    {
        // W0 = I(2), A = [[1,2]], B = [[1],[3]], scaling 2  =>  W' = I + 2 * [[1,2],[3,6]]
        var w0 = eye(2);
        var a = tensor(new float[] { 1, 2 }, [1, 2]);
        var b = tensor(new float[] { 1, 3 }, [2, 1]);

        var merged = LoraMerge.Merge(w0, a, b, 2.0);

        Assert.Equal(new float[] { 3, 4, 6, 13 }, merged.data<float>().ToArray());
    }

    [Fact]
    public void Forgetting_the_scaling_changes_the_answer()
    {
        var w0 = zeros(2, 2);
        var a = tensor(new float[] { 1, 2 }, [1, 2]);
        var b = tensor(new float[] { 1, 3 }, [2, 1]);

        var scaled = LoraMerge.Merge(w0, a, b, 2.0);
        var unscaled = LoraMerge.Merge(w0, a, b, 1.0);

        Assert.True((scaled - unscaled).abs().max().item<float>() > 0);
    }

    [Fact]
    public void Zero_lora_B_is_inert_and_merges_to_the_base()
    {
        var w0 = randn(4, 6);
        var a = randn(2, 6);
        var b = zeros(4, 2);

        Assert.True(LoraMerge.IsInert(b));
        Assert.Equal(0f, (LoraMerge.Merge(w0, a, b, 2.0) - w0).abs().max().item<float>());
        Assert.False(LoraMerge.IsInert(randn(4, 2)));
    }

    [Fact]
    public void Mismatched_shapes_are_rejected()
        => Assert.Throws<ArgumentException>(() => LoraMerge.Merge(zeros(4, 6), zeros(2, 6), zeros(4, 3), 1.0));
}
