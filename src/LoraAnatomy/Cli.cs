using System.Text.Json;
using static TorchSharp.torch;

namespace LoraAnatomy;

/// <summary>The <c>check</c> and <c>merge</c> subcommands, over a PEFT adapter directory.</summary>
public static class Cli
{
    public static int Run(string[] args)
    {
        try
        {
            return args[0] switch
            {
                "check" when args.Length == 2 => Check(args[1]),
                "merge" when args.Length == 4 => Merge(args[1], args[2], args[3]),
                _ => Usage(),
            };
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or KeyNotFoundException
                                       or NotSupportedException or ArgumentException or JsonException)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 2;
        }
    }

    private static int Usage()
    {
        Console.Error.WriteLine("usage:\n  check <adapter_dir>\n  merge <adapter_dir> <base.safetensors> <out.safetensors>");
        return 2;
    }

    private static (Dictionary<string, Tensor> Adapter, double Scaling, int R, int Alpha, bool RsLora) LoadAdapter(string dir)
    {
        string weights = Path.Combine(dir, "adapter_model.safetensors");
        string config = Path.Combine(dir, "adapter_config.json");
        if (!File.Exists(weights) || !File.Exists(config))
            throw new FileNotFoundException($"'{dir}' must contain adapter_model.safetensors and adapter_config.json.");

        using var doc = JsonDocument.Parse(File.ReadAllText(config));
        int r = doc.RootElement.GetProperty("r").GetInt32();
        int alpha = doc.RootElement.GetProperty("lora_alpha").GetInt32();
        bool rs = doc.RootElement.TryGetProperty("use_rslora", out var v) && v.ValueKind == JsonValueKind.True;
        return (SafeTensors.Load(weights), LoraMerge.Scaling(r, alpha, rs), r, alpha, rs);
    }

    /// <summary>Exit 0 when trained, 1 when every lora_B is zero (a guaranteed no-op), 2 on error.</summary>
    private static int Check(string dir)
    {
        var (adapter, scaling, r, alpha, rs) = LoadAdapter(dir);
        var bs = adapter.Where(kv => kv.Key.EndsWith(".lora_B.weight", StringComparison.Ordinal)).ToList();
        if (bs.Count == 0)
            throw new InvalidDataException("no lora_B tensors found: not a LoRA adapter.");

        int inert = bs.Count(kv => LoraMerge.IsInert(kv.Value));
        Console.WriteLine($"r={r} lora_alpha={alpha} use_rslora={rs} scaling={scaling}");
        Console.WriteLine($"{adapter.Count} tensors, {adapter.Values.Sum(t => t.numel()):N0} parameters, {bs.Count} lora_B matrices");
        Console.WriteLine($"{inert} of {bs.Count} lora_B matrices are exactly zero");

        if (inert == bs.Count)
        {
            Console.WriteLine("INERT: merging this adapter returns the base model unchanged. It was never trained, or training did not move it.");
            return 1;
        }

        Console.WriteLine(inert > 0
            ? "TRAINED (partially): some modules are untouched."
            : "TRAINED: every module carries a non-zero update.");
        return 0;
    }

    private static int Merge(string dir, string basePath, string outPath)
    {
        var (adapter, scaling, _, _, _) = LoadAdapter(dir);
        if (adapter.Where(kv => kv.Key.EndsWith(".lora_B.weight", StringComparison.Ordinal)).All(kv => LoraMerge.IsInert(kv.Value)))
            throw new InvalidDataException("refusing to merge: every lora_B is zero, so the output would equal the base. Run 'check' for details.");

        var baseWeights = SafeTensors.Load(basePath);
        var (merged, patched) = LoraMerge.MergeCheckpoint(baseWeights, adapter, scaling);
        SafeTensorsWriter.Save(outPath, merged);
        Console.WriteLine($"patched {patched.Count} of {merged.Count} tensors (scaling {scaling}); wrote {outPath}");
        return 0;
    }
}
