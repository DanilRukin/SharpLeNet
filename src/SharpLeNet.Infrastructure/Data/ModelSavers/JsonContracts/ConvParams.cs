using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;

/// <summary>
/// DTO для параметров сверточного слоя
/// </summary>
public class ConvParams
{
    [JsonPropertyName("inputChannels")]
    public int InputChannels { get; set; }

    [JsonPropertyName("outputChannels")]
    public int OutputChannels { get; set; }

    [JsonPropertyName("kernelSize")]
    public int KernelSize { get; set; }

    [JsonPropertyName("stride")]
    public int Stride { get; set; }

    [JsonPropertyName("padding")]
    public int Padding { get; set; }
}
