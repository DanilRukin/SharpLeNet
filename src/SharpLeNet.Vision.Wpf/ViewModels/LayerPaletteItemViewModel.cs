using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class LayerPaletteItemViewModel : BaseViewModel
{
    public string Name { get; set; } = string.Empty;
    public string Subtitle { get; set; } = string.Empty;
    public string Badge { get; set; } = string.Empty;
    public string Color { get; set; } = string.Empty;
    public string BorderColor { get; set; } = string.Empty;
    public string BackgroundGradient { get; set; } = string.Empty;
    public string Icon { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public (string Label, string Value)[] DefaultParams { get; set; } = System.Array.Empty<(string, string)>();
    public LayerType LayerType { get; set; }
}
