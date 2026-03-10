using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class StatusBarViewModel : BaseViewModel
{
    private string _computeDevice = "GPU (CUDA)";
    private double _vramUsed = 3.2;
    private double _vramTotal = 8.0;
    private string _currentEpoch = "04/15";
    private int _currentBatch = 128;
    private int _totalBatches = 469;
    private double _learningRate = 0.001;
    private TimeSpan _eta = TimeSpan.FromMinutes(12).Add(TimeSpan.FromSeconds(48));
    private double _currentLoss = 0.284;
    private bool _isComputing = true;

    public string ComputeDevice
    {
        get => _computeDevice;
        set => SetProperty(ref _computeDevice, value);
    }

    public double VramUsed
    {
        get => _vramUsed;
        set => SetProperty(ref _vramUsed, value);
    }

    public double VramTotal
    {
        get => _vramTotal;
        set => SetProperty(ref _vramTotal, value);
    }

    public double VramPercentage => (VramUsed / VramTotal) * 100;

    public string CurrentEpoch
    {
        get => _currentEpoch;
        set => SetProperty(ref _currentEpoch, value);
    }

    public int CurrentBatch
    {
        get => _currentBatch;
        set => SetProperty(ref _currentBatch, value);
    }

    public int TotalBatches
    {
        get => _totalBatches;
        set => SetProperty(ref _totalBatches, value);
    }

    public double BatchProgress => (double)CurrentBatch / TotalBatches * 100;

    public double LearningRate
    {
        get => _learningRate;
        set => SetProperty(ref _learningRate, value);
    }

    public TimeSpan Eta
    {
        get => _eta;
        set => SetProperty(ref _eta, value);
    }

    public string EtaFormatted => _eta.ToString(@"hh\:mm\:ss");

    public double CurrentLoss
    {
        get => _currentLoss;
        set => SetProperty(ref _currentLoss, value);
    }

    public bool IsComputing
    {
        get => _isComputing;
        set => SetProperty(ref _isComputing, value);
    }

    public void UpdateTrainingStatus(string status, int epoch, int batch)
    {
        // Update training-related properties
        IsComputing = status == "Training";
    }

    public void UpdateMemoryUsage(double used, double total)
    {
        VramUsed = used;
        VramTotal = total;
    }
}
