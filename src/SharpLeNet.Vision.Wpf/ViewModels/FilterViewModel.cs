using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class FilterViewModel : BaseViewModel
{
    private int _index;
    private string _size = string.Empty;
    private double[,]? _weights;
    private double _minWeight;
    private double _maxWeight;
    private double _meanWeight;

    public int Index
    {
        get => _index;
        set => SetProperty(ref _index, value);
    }

    public string Size
    {
        get => _size;
        set => SetProperty(ref _size, value);
    }

    public double[,]? Weights
    {
        get => _weights;
        set => SetProperty(ref _weights, value);
    }

    public double MinWeight
    {
        get => _minWeight;
        set => SetProperty(ref _minWeight, value);
    }

    public double MaxWeight
    {
        get => _maxWeight;
        set => SetProperty(ref _maxWeight, value);
    }

    public double MeanWeight
    {
        get => _meanWeight;
        set => SetProperty(ref _meanWeight, value);
    }
}
