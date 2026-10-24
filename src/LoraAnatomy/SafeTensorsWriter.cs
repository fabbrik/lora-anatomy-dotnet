using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace LoraAnatomy;

/// <summary>Writes float32 tensors as a safetensors file; the inverse of <see cref="SafeTensors.Load"/>.</summary>
public static class SafeTensorsWriter
{
    public static void Save(string path, IReadOnlyDictionary<string, Tensor> tensors)
    {
        // Names are sorted so the same tensors always produce byte-identical files.
        var names = tensors.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();

        var header = new SortedDictionary<string, object>(StringComparer.Ordinal);
        var blobs = new List<byte[]>();
        long offset = 0;
        foreach (string name in names)
        {
            Tensor t = tensors[name].to(ScalarType.Float32).contiguous();
            float[] values = t.data<float>().ToArray();
            byte[] raw = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, raw, 0, raw.Length);
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = "F32",
                ["shape"] = t.shape,
                ["data_offsets"] = new[] { offset, offset + raw.Length },
            };
            blobs.Add(raw);
            offset += raw.Length;
        }

        header["__metadata__"] = new Dictionary<string, string> { ["format"] = "pt" };
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));

        using var file = File.Create(path);
        file.Write(BitConverter.GetBytes((ulong)json.Length));
        file.Write(json);
        foreach (byte[] blob in blobs) file.Write(blob);
    }
}
