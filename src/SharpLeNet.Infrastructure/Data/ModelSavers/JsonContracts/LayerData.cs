using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// DTO для слоя
/// </summary>
public class LayerData
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("parameters")]
    public List<TensorData> Parameters { get; set; } = new();

    [JsonPropertyName("linearParams")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LinearParams? LinearParams { get; set; }

    [JsonPropertyName("convParams")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ConvParams? ConvParams { get; set; }
}
