using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// DTO для параметров линейного слоя
/// </summary>
public class LinearParams
{
    [JsonPropertyName("inputSize")]
    public int InputSize { get; set; }

    [JsonPropertyName("outputSize")]
    public int OutputSize { get; set; }
}
