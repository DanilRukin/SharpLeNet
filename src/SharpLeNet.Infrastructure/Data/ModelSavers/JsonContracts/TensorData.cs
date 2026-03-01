using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// DTO для сохранения тензора
/// </summary>
public class TensorData
{
    [JsonPropertyName("shape")]
    public int[] Shape { get; set; } = Array.Empty<int>();

    [JsonPropertyName("data")]
    public double[] Data { get; set; } = Array.Empty<double>();
}
