using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class TopClassViewModel : BaseViewModel
{
    private string _classIndex;
    private string _className;
    private double _probability;

    public string ClassIndex
    {
        get => _classIndex;
        set => SetProperty(ref _classIndex, value);
    }

    public string ClassName
    {
        get => _className;
        set => SetProperty(ref _className, value);
    }

    public double Probability
    {
        get => _probability;
        set => SetProperty(ref _probability, value);
    }
}
