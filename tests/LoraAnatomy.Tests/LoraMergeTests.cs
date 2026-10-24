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

public sealed class CheckpointMergeTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("lora-anatomy-ckpt-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private const string Module = "base_model.model.model.layers.0.self_attn.q_proj";
    private const string Key = "model.layers.0.self_attn.q_proj.weight";

    private static Dictionary<string, Tensor> Adapter(Tensor b) => new()
    {
        [$"{Module}.lora_A.weight"] = tensor(new float[] { 1, 2 }, [1, 2]),
        [$"{Module}.lora_B.weight"] = b,
    };

    [Fact]
    public void Base_key_drops_the_peft_prefix_and_adds_weight()
        => Assert.Equal(Key, LoraMerge.BaseKeyFor(Module));

    [Fact]
    public void Patches_targeted_modules_and_passes_the_rest_through()
    {
        var other = ones(3, 3);
        var baseWeights = new Dictionary<string, Tensor> { [Key] = eye(2), ["model.norm.weight"] = other };

        var (merged, patched) = LoraMerge.MergeCheckpoint(baseWeights, Adapter(tensor(new float[] { 1, 3 }, [2, 1])), 2.0);

        Assert.Equal([Key], patched);
        Assert.Equal(new float[] { 3, 4, 6, 13 }, merged[Key].data<float>().ToArray());
        Assert.Same(other, merged["model.norm.weight"]);
    }

    [Fact]
    public void A_module_the_base_lacks_is_an_error_not_a_skip()
    {
        var baseWeights = new Dictionary<string, Tensor> { ["model.norm.weight"] = ones(2) };

        var ex = Assert.Throws<KeyNotFoundException>(() =>
            LoraMerge.MergeCheckpoint(baseWeights, Adapter(ones(2, 1)), 1.0));

        Assert.Contains(Key, ex.Message);
    }

    [Fact]
    public void Written_checkpoint_reads_back_identically()
    {
        var tensors = new Dictionary<string, Tensor> { ["a"] = randn(3, 4), ["b.weight"] = randn(5) };
        string path = Path.Combine(_dir, "out.safetensors");

        SafeTensorsWriter.Save(path, tensors);
        var back = SafeTensors.Load(path);

        Assert.Equal(tensors.Keys.Order(), back.Keys.Order());
        foreach (var (name, t) in tensors)
        {
            Assert.Equal(t.shape, back[name].shape);
            Assert.Equal(0f, (t - back[name]).abs().max().item<float>());
        }
    }

    [Fact]
    public void Same_tensors_write_byte_identical_files()
    {
        var tensors = new Dictionary<string, Tensor> { ["z"] = ones(2), ["a"] = zeros(2) };
        string one = Path.Combine(_dir, "1.safetensors"), two = Path.Combine(_dir, "2.safetensors");

        SafeTensorsWriter.Save(one, tensors);
        SafeTensorsWriter.Save(two, tensors);

        Assert.Equal(File.ReadAllBytes(one), File.ReadAllBytes(two));
    }
}
