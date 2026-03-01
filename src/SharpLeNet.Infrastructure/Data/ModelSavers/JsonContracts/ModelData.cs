using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// DTO для всей модели
/// </summary>
public class ModelData
{
    [JsonPropertyName("version")]
    public int Version { get; set; } = 1;

    [JsonPropertyName("createdAt")]
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    [JsonPropertyName("layers")]
    public List<LayerData> Layers { get; set; } = new();

    [JsonPropertyName("metadata")]
    public Dictionary<string, string> Metadata { get; set; } = new();
}
