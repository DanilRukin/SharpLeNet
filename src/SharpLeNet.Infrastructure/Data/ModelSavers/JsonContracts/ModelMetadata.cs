namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// Метаданные модели (без весов)
/// </summary>
public class ModelMetadata
{
    public string Identifier { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public int TotalLayers { get; set; }
    public Dictionary<string, string> Metadata { get; set; } = new();
}
