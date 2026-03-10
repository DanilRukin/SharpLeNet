using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class FeatureMapViewModel : BaseViewModel
{
    private int _index;
    private string _name = string.Empty;
    private double _minValue;
    private double _maxValue;
    private double _meanValue;
    private double[,]? _heatmapData;

    public int Index
    {
        get => _index;
        set => SetProperty(ref _index, value);
    }

    public string Name
    {
        get => _name;
        set => SetProperty(ref _name, value);
    }

    public double MinValue
    {
        get => _minValue;
        set => SetProperty(ref _minValue, value);
    }

    public double MaxValue
    {
        get => _maxValue;
        set => SetProperty(ref _maxValue, value);
    }

    public double MeanValue
    {
        get => _meanValue;
        set => SetProperty(ref _meanValue, value);
    }

    public double[,]? HeatmapData
    {
        get => _heatmapData;
        set => SetProperty(ref _heatmapData, value);
    }
}
