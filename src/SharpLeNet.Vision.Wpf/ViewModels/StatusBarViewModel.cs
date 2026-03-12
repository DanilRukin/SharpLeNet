using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows;
using System.Windows.Media;

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
    private List<double> _lossHistory = new List<double>();

    public StatusBarViewModel()
    {
        // Инициализация истории лосса для спарклайна
        var random = new Random(42);
        for (int i = 0; i < 20; i++)
        {
            _lossHistory.Add(0.8 - (i * 0.03) + (random.NextDouble() * 0.05 - 0.025));
        }
        UpdateSparklinePoints();
    }

    // Свойства
    public string ComputeDevice
    {
        get => _computeDevice;
        set => SetProperty(ref _computeDevice, value);
    }

    public double VramUsed
    {
        get => _vramUsed;
        set
        {
            if (SetProperty(ref _vramUsed, value))
            {
                OnPropertyChanged(nameof(VramDisplay));
                OnPropertyChanged(nameof(VramPercentage));
            }
        }
    }

    public double VramTotal
    {
        get => _vramTotal;
        set
        {
            if (SetProperty(ref _vramTotal, value))
            {
                OnPropertyChanged(nameof(VramDisplay));
                OnPropertyChanged(nameof(VramPercentage));
            }
        }
    }

    public string VramDisplay => $"{VramUsed:F1} / {VramTotal:F0} GB";
    public double VramPercentage => (VramUsed / VramTotal) * 100;

    public string CurrentEpoch
    {
        get => _currentEpoch;
        set
        {
            if (SetProperty(ref _currentEpoch, value))
            {
                OnPropertyChanged(nameof(EpochDisplay));
            }
        }
    }

    public string EpochDisplay => $"epoch {CurrentEpoch}";

    public int CurrentBatch
    {
        get => _currentBatch;
        set
        {
            if (SetProperty(ref _currentBatch, value))
            {
                OnPropertyChanged(nameof(BatchDisplay));
                OnPropertyChanged(nameof(BatchProgress));
            }
        }
    }

    public int TotalBatches
    {
        get => _totalBatches;
        set
        {
            if (SetProperty(ref _totalBatches, value))
            {
                OnPropertyChanged(nameof(BatchDisplay));
                OnPropertyChanged(nameof(BatchProgress));
            }
        }
    }

    public string BatchDisplay => $"batch {CurrentBatch}/{TotalBatches}";
    public double BatchProgress => (double)CurrentBatch / TotalBatches * 100;

    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (SetProperty(ref _learningRate, value))
            {
                OnPropertyChanged(nameof(LearningRateDisplay));
            }
        }
    }

    public string LearningRateDisplay => LearningRate.ToString("e2");

    public TimeSpan Eta
    {
        get => _eta;
        set
        {
            if (SetProperty(ref _eta, value))
            {
                OnPropertyChanged(nameof(EtaDisplay));
            }
        }
    }

    public string EtaDisplay => _eta.ToString(@"hh\:mm\:ss");

    public double CurrentLoss
    {
        get => _currentLoss;
        set
        {
            if (SetProperty(ref _currentLoss, value))
            {
                OnPropertyChanged(nameof(CurrentLossDisplay));
                UpdateLossHistory(value);
            }
        }
    }

    public string CurrentLossDisplay => _currentLoss.ToString("F3");

    public bool IsComputing
    {
        get => _isComputing;
        set => SetProperty(ref _isComputing, value);
    }

    // Данные для спарклайна
    public PointCollection SparklinePoints { get; private set; } = new PointCollection();
    public PointCollection SparklineFillPoints { get; private set; } = new PointCollection();

    // Методы обновления
    public void UpdateTrainingStatus(int epoch, int batch, double loss, double learningRate, TimeSpan? eta = null)
    {
        CurrentEpoch = $"{epoch:D2}/{15:D2}"; // 15 - total epochs
        CurrentBatch = batch;
        CurrentLoss = loss;
        LearningRate = learningRate;

        if (eta.HasValue)
            Eta = eta.Value;
    }

    public void UpdateMemoryUsage(double used, double total)
    {
        VramUsed = used;
        VramTotal = total;
    }

    private void UpdateLossHistory(double newLoss)
    {
        _lossHistory.Add(newLoss);
        if (_lossHistory.Count > 20)
            _lossHistory.RemoveAt(0);

        UpdateSparklinePoints();
    }

    private void UpdateSparklinePoints()
    {
        if (_lossHistory.Count == 0)
            return;

        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (var value in _lossHistory)
        {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        double range = max - min;
        if (range < 0.01) range = 0.01;

        var points = new PointCollection();
        var fillPoints = new PointCollection();

        double width = 100;
        double height = 30;
        double step = width / (_lossHistory.Count - 1);

        for (int i = 0; i < _lossHistory.Count; i++)
        {
            double x = i * step;
            double y = height - ((_lossHistory[i] - min) / range * height * 0.8 + height * 0.1);

            points.Add(new Point(x, y));
            fillPoints.Add(new Point(x, y));
        }

        // Добавляем точки для заливки
        fillPoints.Add(new Point(width, height + 5));
        fillPoints.Add(new Point(0, height + 5));

        SparklinePoints = points;
        SparklineFillPoints = fillPoints;

        OnPropertyChanged(nameof(SparklinePoints));
        OnPropertyChanged(nameof(SparklineFillPoints));
    }

    // Сброс состояния
    public void Reset()
    {
        CurrentEpoch = "00/15";
        CurrentBatch = 0;
        CurrentLoss = 0;
        LearningRate = 0.001;
        Eta = TimeSpan.Zero;
        _lossHistory.Clear();
        UpdateSparklinePoints();
    }
}
