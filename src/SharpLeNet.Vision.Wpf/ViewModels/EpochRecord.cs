using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class EpochRecord : BaseViewModel
{
    private int _epoch;
    private double _trainLoss;
    private double _valLoss;
    private double _trainAcc;
    private double _valAcc;
    private string _time = string.Empty;
    private double _learningRate;

    public int Epoch
    {
        get => _epoch;
        set => SetProperty(ref _epoch, value);
    }

    public double TrainLoss
    {
        get => _trainLoss;
        set => SetProperty(ref _trainLoss, value);
    }

    public double ValLoss
    {
        get => _valLoss;
        set => SetProperty(ref _valLoss, value);
    }

    public double TrainAcc
    {
        get => _trainAcc;
        set => SetProperty(ref _trainAcc, value);
    }

    public double ValAcc
    {
        get => _valAcc;
        set => SetProperty(ref _valAcc, value);
    }

    public string Time
    {
        get => _time;
        set => SetProperty(ref _time, value);
    }

    public double LearningRate
    {
        get => _learningRate;
        set => SetProperty(ref _learningRate, value);
    }

    public string TrainLossDisplay => TrainLoss.ToString("F4");
    public string ValLossDisplay => ValLoss.ToString("F4");
    public string TrainAccDisplay => TrainAcc.ToString("F2") + "%";
    public string ValAccDisplay => ValAcc.ToString("F2") + "%";
    public string LearningRateDisplay => LearningRate.ToString("e2");
}