using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class LayerBlockViewModel : BaseViewModel
{
    private Point _position;
    private bool _isSelected;
    private string _id = string.Empty;
    private string _name = string.Empty;
    private string _subtitle = string.Empty;
    private string _type = string.Empty;
    private string _color = string.Empty;
    private string _borderColor = string.Empty;
    private string _backgroundGradient = string.Empty;
    private string _outputDim = string.Empty;
    private string _icon = string.Empty;

    public LayerBlockViewModel()
    {
        Parameters = new ObservableCollection<ParameterViewModel>();
    }

    public string Id
    {
        get => _id;
        set => SetProperty(ref _id, value);
    }

    public string Name
    {
        get => _name;
        set => SetProperty(ref _name, value);
    }

    public string Subtitle
    {
        get => _subtitle;
        set => SetProperty(ref _subtitle, value);
    }

    public string Type
    {
        get => _type;
        set => SetProperty(ref _type, value);
    }

    public string Color
    {
        get => _color;
        set => SetProperty(ref _color, value);
    }

    public string BorderColor
    {
        get => _borderColor;
        set => SetProperty(ref _borderColor, value);
    }

    public string BackgroundGradient
    {
        get => _backgroundGradient;
        set => SetProperty(ref _backgroundGradient, value);
    }

    public string OutputDim
    {
        get => _outputDim;
        set => SetProperty(ref _outputDim, value);
    }

    public string Icon
    {
        get => _icon;
        set => SetProperty(ref _icon, value);
    }

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

    public ObservableCollection<ParameterViewModel> Parameters { get; set; }
}