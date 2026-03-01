namespace SharpLeNet.Infrastructure.Data.ModelSavers.DbContracts;

/// <summary>
/// Модель для таблицы Tensors
/// </summary>
public class DbTensor
{
    public int Id { get; set; }
    public int LayerId { get; set; }
    public int ParameterIndex { get; set; }
    public string ShapeJson { get; set; } = string.Empty; // [1, 28, 28] в JSON
    public byte[] DataBlob { get; set; } = Array.Empty<byte>(); // бинарные данные
    public int DataLength { get; set; }
}
