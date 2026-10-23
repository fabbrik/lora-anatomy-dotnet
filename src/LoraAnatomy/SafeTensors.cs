using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace LoraAnatomy;

/// <summary>
/// Minimal safetensors reader.
///
/// Layout: [8 bytes little-endian u64 = header length][JSON header][raw tensor data]
/// The JSON header maps tensor name -> { dtype, shape, data_offsets: [start, end] },
/// where offsets are relative to the start of the data block. Data is C-contiguous
/// (row-major).
/// </summary>
public static class SafeTensors
{
    public static Dictionary<string, Tensor> Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        ulong headerLen = reader.ReadUInt64();
        if (headerLen > (ulong)(stream.Length - 8))
            throw new InvalidDataException(
                $"safetensors header length {headerLen} exceeds file size {stream.Length}: not a safetensors file, or truncated.");
        string headerJson = Encoding.UTF8.GetString(reader.ReadBytes((int)headerLen));

        long dataStart = 8 + (long)headerLen;
        var result = new Dictionary<string, Tensor>();

        using var doc = JsonDocument.Parse(headerJson);
        foreach (var entry in doc.RootElement.EnumerateObject())
        {
            // "__metadata__" is a reserved key holding free-form strings, not a tensor.
            if (entry.Name == "__metadata__") continue;

            string dtype = entry.Value.GetProperty("dtype").GetString()!;
            long[] shape = entry.Value.GetProperty("shape")
                .EnumerateArray().Select(e => e.GetInt64()).ToArray();
            var offsets = entry.Value.GetProperty("data_offsets")
                .EnumerateArray().Select(e => e.GetInt64()).ToArray();

            long byteCount = offsets[1] - offsets[0];
            stream.Seek(dataStart + offsets[0], SeekOrigin.Begin);
            byte[] raw = reader.ReadBytes((int)byteCount);

            result[entry.Name] = ToTensor(raw, dtype, shape);
        }

        return result;
    }

    private static Tensor ToTensor(byte[] raw, string dtype, long[] shape)
    {
        switch (dtype)
        {
            case "F32":
            {
                var values = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
                return tensor(values, shape);
            }
            case "F64":
            {
                var values = new double[raw.Length / 8];
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
                return tensor(values, shape);
            }
            case "F16":
            {
                // IEEE half -> float32. Decoded by hand: torch.tensor(short[]) would
                // reinterpret nothing, it would just cast the integer bit patterns.
                var values = new float[raw.Length / 2];
                for (int i = 0; i < values.Length; i++)
                    values[i] = (float)BitConverter.ToHalf(raw, i * 2);
                return tensor(values, shape);
            }
            case "BF16":
            {
                // bfloat16 is the top 16 bits of a float32, so widening is a shift.
                // Most adapters trained on recent GPUs are stored this way.
                var values = new float[raw.Length / 2];
                for (int i = 0; i < values.Length; i++)
                    values[i] = BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(raw, i * 2) << 16);
                return tensor(values, shape);
            }
            default:
                throw new NotSupportedException(
                    $"dtype '{dtype}' not handled yet. Add it to SafeTensors.ToTensor.");
        }
    }
}
