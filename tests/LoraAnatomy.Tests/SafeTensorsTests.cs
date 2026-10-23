using System.Text;
using LoraAnatomy;
using Xunit;

namespace LoraAnatomy.Tests;

public sealed class SafeTensorsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("lora-anatomy-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>Write a one-tensor safetensors file by hand, independent of the reader under test.</summary>
    private string Write(string dtype, long[] shape, byte[] data)
    {
        string header = $"{{\"__metadata__\":{{\"format\":\"pt\"}},\"t\":{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[0,{data.Length}]}}}}";
        byte[] h = Encoding.UTF8.GetBytes(header);
        string path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".safetensors");
        using var f = File.Create(path);
        f.Write(BitConverter.GetBytes((ulong)h.Length));
        f.Write(h);
        f.Write(data);
        return path;
    }

    [Fact]
    public void F32_roundtrips_and_skips_metadata()
    {
        float[] v = [1.5f, -2f, 0.25f, 8f];
        var bytes = new byte[16];
        Buffer.BlockCopy(v, 0, bytes, 0, 16);

        var t = SafeTensors.Load(Write("F32", [2, 2], bytes));

        Assert.Single(t);
        Assert.Equal(new long[] { 2, 2 }, t["t"].shape);
        Assert.Equal(v, t["t"].data<float>().ToArray());
    }

    [Fact]
    public void F16_is_decoded_not_cast()
    {
        Half[] v = [(Half)1.5f, (Half)(-2f), (Half)0.25f];
        var bytes = v.SelectMany(h => BitConverter.GetBytes(h)).ToArray();

        var t = SafeTensors.Load(Write("F16", [3], bytes));

        Assert.Equal(new[] { 1.5f, -2f, 0.25f }, t["t"].data<float>().ToArray());
    }

    [Fact]
    public void BF16_widens_to_float32()
    {
        // bfloat16 = top 16 bits of the float32 pattern; these values are exactly representable.
        float[] v = [1.5f, -2f, 0.25f];
        var bytes = v.SelectMany(f => BitConverter.GetBytes((ushort)(BitConverter.SingleToUInt32Bits(f) >> 16))).ToArray();

        var t = SafeTensors.Load(Write("BF16", [3], bytes));

        Assert.Equal(v, t["t"].data<float>().ToArray());
    }

    [Fact]
    public void Unsupported_dtype_fails_loudly()
        => Assert.Throws<NotSupportedException>(() => SafeTensors.Load(Write("I8", [1], [1])));

    [Fact]
    public void Truncated_header_is_rejected()
    {
        string path = Path.Combine(_dir, "bad.safetensors");
        File.WriteAllBytes(path, [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 1, 2]);
        Assert.Throws<InvalidDataException>(() => SafeTensors.Load(path));
    }
}
