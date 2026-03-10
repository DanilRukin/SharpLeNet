using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class LayerBlockViewModel : BaseViewModel
{
    private Point _position;
    private bool _isSelected;

    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Subtitle { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public string Color { get; set; } = string.Empty;
    public string BorderColor { get; set; } = string.Empty;
    public string BackgroundGradient { get; set; } = string.Empty;
    public string OutputDim { get; set; } = string.Empty;
    public string Icon { get; set; } = string.Empty;

    public ObservableCollection<ParameterViewModel> Parameters { get; set; } = new();

    public Point Position
    {
        get => _position;
        set => SetProperty(ref _position, value);
    }

    public bool IsSelected
    {
        get => _isSelected;
        set => SetProperty(ref _isSelected, value);
    }
}
