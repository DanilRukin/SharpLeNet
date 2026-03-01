namespace SharpLeNet.Infrastructure.Data.ModelSavers.DbContracts;

/// <summary>
/// Модель для таблицы Layers
/// </summary>
public class DbLayer
{
    public int Id { get; set; }
    public int ModelId { get; set; }
    public int LayerIndex { get; set; }
    public string LayerType { get; set; } = string.Empty;

    // Параметры линейного слоя
    public int? InputSize { get; set; }
    public int? OutputSize { get; set; }

    // Параметры сверточного слоя
    public int? InputChannels { get; set; }
    public int? OutputChannels { get; set; }
    public int? KernelSize { get; set; }
    public int? Stride { get; set; }
    public int? Padding { get; set; }
}
