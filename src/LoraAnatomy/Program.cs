using System.Text.Json;
using LoraAnatomy;
using static TorchSharp.torch;

// ---------------------------------------------------------------------------
// lora-anatomy-dotnet
//
// Load a real PEFT LoRA adapter, inspect it, merge it into the base weight,
// and check the result against Python PEFT's own merge.
//
//     W' = W0 + (alpha / r) * B @ A
// ---------------------------------------------------------------------------

// "synthetic" -> reference/out      (toy model, built offline by make_reference.py)
// "real"      -> reference/out-real (published Hub adapter, make_real_reference.py)
string which = args.FirstOrDefault() ?? "synthetic";
string root = Path.Combine(AppContext.BaseDirectory, "../../../../..");
string refDir = Path.GetFullPath(Path.Combine(root,
    which == "real" ? "reference/out-real" : "reference/out"));

// The synthetic run nests the adapter one level down; the real run sits flat.
string adapterDir = Directory.Exists(Path.Combine(refDir, "adapter"))
    ? Path.Combine(refDir, "adapter")
    : refDir;

string adapterPath = Path.Combine(adapterDir, "adapter_model.safetensors");
string configPath = Path.Combine(adapterDir, "adapter_config.json");
string truthPath = Path.Combine(refDir, "ground_truth.safetensors");

if (!File.Exists(adapterPath))
{
    Console.Error.WriteLine($"Missing {adapterPath}\nRun: cd reference && uv run " +
        (which == "real" ? "make_real_reference.py" : "make_reference.py"));
    return 1;
}

Console.WriteLine($"=== dataset: {which} ({refDir}) ===\n");

// --- 1. adapter config ------------------------------------------------------
using var cfgDoc = JsonDocument.Parse(File.ReadAllText(configPath));
int r = cfgDoc.RootElement.GetProperty("r").GetInt32();
int alpha = cfgDoc.RootElement.GetProperty("lora_alpha").GetInt32();

// GOTCHA: older adapters have no "use_rslora" key at all -- it must default to
// false rather than being read blindly. With rsLoRA the divisor is sqrt(r).
bool useRsLora = cfgDoc.RootElement.TryGetProperty("use_rslora", out var rs) && rs.GetBoolean();
double scaling = LoraMerge.Scaling(r, alpha, useRsLora);

Console.WriteLine("=== adapter_config.json ===");
Console.WriteLine($"  r            = {r}");
Console.WriteLine($"  lora_alpha   = {alpha}");
Console.WriteLine($"  use_rslora   = {useRsLora}{(cfgDoc.RootElement.TryGetProperty("use_rslora", out _) ? "" : "   (key absent -- defaulted)")}");
Console.WriteLine($"  scaling      = {(useRsLora ? "alpha/sqrt(r)" : "alpha/r")} = {scaling}");
if (Math.Abs(scaling - 1.0) < 1e-12)
    Console.WriteLine("  NOTE: scaling == 1, so forgetting it is INVISIBLE on this adapter.");
Console.WriteLine();

// --- 2. what is actually in the adapter file --------------------------------
var adapter = SafeTensors.Load(adapterPath);

Console.WriteLine("=== adapter tensors ===");
foreach (var (name, t) in adapter.OrderBy(kv => kv.Key).Take(6))
    Console.WriteLine($"  {name,-72} [{string.Join(",", t.shape)}]");
if (adapter.Count > 6) Console.WriteLine($"  ... and {adapter.Count - 6} more");

long trainable = adapter.Values.Sum(t => t.numel());
Console.WriteLine($"\n  {adapter.Count} tensors, {trainable:N0} trainable parameters total");
Console.WriteLine();

// --- 3. merge every targeted module and verify against PEFT ------------------
var truth = SafeTensors.Load(truthPath);

// Adapter keys carry PEFT's wrapper prefix; ground-truth keys use the plain
// module path. Strip the prefix to line them up. (See GOTCHAS: "model" x3.)
const string prefix = "base_model.model.";

// Only verify modules the reference actually dumped -- a real adapter may touch
// hundreds, and dumping every base weight would be gigabytes.
var targets = adapter.Keys
    .Where(k => k.EndsWith(".lora_A.weight"))
    .Select(k => k[..^".lora_A.weight".Length])
    .Where(k => truth.ContainsKey($"{(k.StartsWith(prefix) ? k[prefix.Length..] : k)}.w0"))
    .OrderBy(k => k)
    .ToList();

Console.WriteLine($"=== merge + verification against Python PEFT ({targets.Count} modules) ===");
Console.WriteLine($"  {"module",-42} {"delta",12} {"W0+BA",12} {"W0+(a/r)BA",12}");

bool allOk = true;
foreach (var target in targets)
{
    var A = adapter[$"{target}.lora_A.weight"]; // (r, k)
    var B = adapter[$"{target}.lora_B.weight"]; // (d, r)
    var ba = B.matmul(A);                       // (d, k)

    string plain = target.StartsWith(prefix) ? target[prefix.Length..] : target;
    var w0 = truth[$"{plain}.w0"];
    var wTruth = truth[$"{plain}.w_merged"];

    double errUnscaled = (w0 + ba - wTruth).abs().max().item<float>();
    double errScaled = (w0 + scaling * ba - wTruth).abs().max().item<float>();
    double deltaMag = (wTruth - w0).abs().max().item<float>();

    bool ok = errScaled < 1e-5;
    allOk &= ok;

    string shortName = plain.Replace("model.layers.", "L").Replace(".self_attn.", ".");
    Console.WriteLine($"  {shortName,-42} {deltaMag,12:E3} {errUnscaled,12:E3} {errScaled,12:E3}{(ok ? "" : "  FAIL")}");
}

Console.WriteLine();
Console.WriteLine(Math.Abs(scaling - 1.0) < 1e-12
    ? "  W0+BA and W0+(a/r)BA agree here only because alpha == r. The bug is\n" +
      "  dormant on this adapter and wakes up the moment you load a different one."
    : "  column 3 is the cost of forgetting alpha/r: same order of magnitude\n" +
      "  as the delta itself, so the model still runs and is simply wrong.");
Console.WriteLine();

// --- 4. does the merge actually change behaviour? ---------------------------
// Numerical equality is not behavioural proof. Push a fixed activation through
// the base projection and the merged one, and see how far the output moves.
{
    var target = targets[0];
    string plain = target[prefix.Length..];
    var A = adapter[$"{target}.lora_A.weight"];
    var B = adapter[$"{target}.lora_B.weight"];
    var w0 = truth[$"{plain}.w0"];
    var wPrime = LoraMerge.Merge(w0, A, B, scaling);

    var x = truth["x"];
    var yBase = x.matmul(w0.T);
    var yMerged = x.matmul(wPrime.T);

    double fwdErr = (yMerged - truth["y_merged"]).abs().max().item<float>();
    double moved = (yMerged - yBase).abs().max().item<float>();
    double baseMag = yBase.abs().max().item<float>();
    var cos = nn.functional.cosine_similarity(yBase, yMerged, dim: 1).item<float>();

    Console.WriteLine($"=== forward pass through {plain.Replace("model.layers.", "L").Replace(".self_attn.", ".")} ===");
    Console.WriteLine($"  C# merged activation vs PEFT's : {fwdErr:E3}   (max abs error)");
    Console.WriteLine($"  |y_base| max                   : {baseMag:E3}");
    Console.WriteLine($"  |y_merged - y_base| max        : {moved:E3}   ({moved / baseMag:P1} of signal)");
    Console.WriteLine($"  cosine(y_base, y_merged)       : {cos:F6}");
    Console.WriteLine();
    Console.WriteLine(moved > 0
        ? "  The projection genuinely moved -- the merge changed behaviour, not just bytes."
        : "  WARNING: output identical. The adapter did nothing.");
    Console.WriteLine();
}

Console.WriteLine(allOk
    ? "MATCH: the C# merge reproduces PEFT's merge on every targeted module."
    : "MISMATCH: at least one module does not agree with PEFT.");

return allOk ? 0 : 1;
