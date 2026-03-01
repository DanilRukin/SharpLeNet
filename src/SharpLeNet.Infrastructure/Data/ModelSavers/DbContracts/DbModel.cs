namespace SharpLeNet.Infrastructure.Data.ModelSavers.DbContracts;

/// <summary>
/// Модель для таблицы Models
/// </summary>
public class DbModel
{
    public int Id { get; set; }
    public string Identifier { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public string? Description { get; set; }
    public int Version { get; set; }
    public int TotalLayers { get; set; }
    public int TotalParameters { get; set; }
}
