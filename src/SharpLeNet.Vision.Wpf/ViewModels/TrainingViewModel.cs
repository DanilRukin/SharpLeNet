using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Timers;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class TrainingViewModel : BaseViewModel
{
    private bool _isTraining;
    private bool _isPaused;
    private int _currentEpoch = 4;
    private int _totalEpochs = 15;
    private int _currentBatch = 128;
    private int _totalBatches = 469;
    private double _currentLoss = 0.284;
    private double _validationLoss = 0.312;
    private double _currentAccuracy = 98.11;
    private double _validationAccuracy = 96.45;
    private double _learningRate = 0.001;
    private TimeSpan _eta = TimeSpan.FromMinutes(12).Add(TimeSpan.FromSeconds(48));
    private string _status = "Running";
    private string _scheduleType = "Exponential";
    private ObservableCollection<EpochRecord> _epochHistory;

    public TrainingViewModel()
    {
        // Инициализация команд
        StartCommand = new RelayCommand(Start, (_) => !IsTraining && !IsPaused);
        PauseCommand = new RelayCommand(Pause, (_) => IsTraining && !IsPaused);
        ResumeCommand = new RelayCommand(Resume, (_) => IsPaused);
        StopCommand = new RelayCommand(Stop, (_) => IsTraining || IsPaused);
        RefreshCommand = new RelayCommand(Refresh);

        // Инициализация данных
        InitializeEpochHistory();
        InitializeGraphData();
        InitializeConfusionMatrix();
    }

    // Команды
    public ICommand StartCommand { get; }
    public ICommand PauseCommand { get; }
    public ICommand ResumeCommand { get; }
    public ICommand StopCommand { get; }
    public ICommand RefreshCommand { get; }

    // Состояние обучения
    public bool IsTraining
    {
        get => _isTraining;
        private set => SetProperty(ref _isTraining, value);
    }

    public bool IsPaused
    {
        get => _isPaused;
        private set => SetProperty(ref _isPaused, value);
    }

    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    // Параметры обучения
    public int CurrentEpoch
    {
        get => _currentEpoch;
        set
        {
            if (SetProperty(ref _currentEpoch, value))
                OnPropertyChanged(nameof(EpochDisplay));
        }
    }

    public int TotalEpochs
    {
        get => _totalEpochs;
        set
        {
            if (SetProperty(ref _totalEpochs, value))
                OnPropertyChanged(nameof(EpochDisplay));
        }
    }

    public string EpochDisplay => $"{CurrentEpoch:D2}/{TotalEpochs:D2}";

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

    public string BatchDisplay => $"{CurrentBatch}/{TotalBatches}";
    public double BatchProgress => (double)CurrentBatch / TotalBatches * 100;

    public double CurrentLoss
    {
        get => _currentLoss;
        set
        {
            if (SetProperty(ref _currentLoss, value))
                OnPropertyChanged(nameof(CurrentLossDisplay));
        }
    }

    public string CurrentLossDisplay => _currentLoss.ToString("F3");

    public double ValidationLoss
    {
        get => _validationLoss;
        set
        {
            if (SetProperty(ref _validationLoss, value))
                OnPropertyChanged(nameof(ValidationLossDisplay));
        }
    }

    public string ValidationLossDisplay => _validationLoss.ToString("F3");

    public double CurrentAccuracy
    {
        get => _currentAccuracy;
        set
        {
            if (SetProperty(ref _currentAccuracy, value))
                OnPropertyChanged(nameof(CurrentAccuracyDisplay));
        }
    }

    public string CurrentAccuracyDisplay => _currentAccuracy.ToString("F2") + "%";

    public double ValidationAccuracy
    {
        get => _validationAccuracy;
        set
        {
            if (SetProperty(ref _validationAccuracy, value))
                OnPropertyChanged(nameof(ValidationAccuracyDisplay));
        }
    }

    public string ValidationAccuracyDisplay => _validationAccuracy.ToString("F2") + "%";

    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (SetProperty(ref _learningRate, value))
                OnPropertyChanged(nameof(LearningRateDisplay));
        }
    }

    public string LearningRateDisplay => _learningRate.ToString("e2");

    public TimeSpan Eta
    {
        get => _eta;
        set
        {
            if (SetProperty(ref _eta, value))
                OnPropertyChanged(nameof(EtaDisplay));
        }
    }

    public string EtaDisplay => _eta.ToString(@"hh\:mm\:ss");

    public string ScheduleType
    {
        get => _scheduleType;
        set => SetProperty(ref _scheduleType, value);
    }

    public ObservableCollection<EpochRecord> EpochHistory => _epochHistory;

    // Данные для графиков
    public ObservableCollection<DataPoint> TrainLossPoints { get; private set; }
    public ObservableCollection<DataPoint> ValLossPoints { get; private set; }
    public ObservableCollection<DataPoint> TrainAccPoints { get; private set; }
    public ObservableCollection<DataPoint> ValAccPoints { get; private set; }

    // Confusion Matrix
    public ConfusionMatrixViewModel ConfusionMatrix { get; private set; }

    // Методы команд
    private void Start(object? parameter)
    {
        IsTraining = true;
        IsPaused = false;
        Status = "Running";
    }

    private void Pause(object? parameter)
    {
        IsPaused = true;
        Status = "Paused";
    }

    private void Resume(object? parameter)
    {
        IsPaused = false;
        Status = "Running";
    }

    private void Stop(object? parameter)
    {
        IsTraining = false;
        IsPaused = false;
        Status = "Stopped";
    }

    private void Refresh(object? parameter)
    {
        // Обновление данных
    }

    private void InitializeEpochHistory()
    {
        _epochHistory = new ObservableCollection<EpochRecord>
            {
                new EpochRecord { Epoch = 4, TrainLoss = 0.2842, ValLoss = 0.3120, TrainAcc = 98.42, ValAcc = 98.11, Time = "02:14", LearningRate = 1.00e-3 },
                new EpochRecord { Epoch = 3, TrainLoss = 0.3421, ValLoss = 0.3892, TrainAcc = 97.21, ValAcc = 96.45, Time = "02:12", LearningRate = 1.05e-3 },
                new EpochRecord { Epoch = 2, TrainLoss = 0.4890, ValLoss = 0.5122, TrainAcc = 94.10, ValAcc = 93.20, Time = "02:15", LearningRate = 1.10e-3 },
                new EpochRecord { Epoch = 1, TrainLoss = 0.7241, ValLoss = 0.7812, TrainAcc = 88.42, ValAcc = 87.12, Time = "02:20", LearningRate = 1.20e-3 },
                new EpochRecord { Epoch = 0, TrainLoss = 1.2402, ValLoss = 1.3204, TrainAcc = 72.15, ValAcc = 68.42, Time = "02:25", LearningRate = 1.30e-3 }
            };
    }

    private void InitializeGraphData()
    {
        TrainLossPoints = new ObservableCollection<DataPoint>();
        ValLossPoints = new ObservableCollection<DataPoint>();
        TrainAccPoints = new ObservableCollection<DataPoint>();
        ValAccPoints = new ObservableCollection<DataPoint>();

        // Добавляем точки для графика лосса
        TrainLossPoints.Add(new DataPoint(0, 1.24));
        TrainLossPoints.Add(new DataPoint(1, 0.72));
        TrainLossPoints.Add(new DataPoint(2, 0.49));
        TrainLossPoints.Add(new DataPoint(3, 0.34));
        TrainLossPoints.Add(new DataPoint(4, 0.28));

        ValLossPoints.Add(new DataPoint(0, 1.32));
        ValLossPoints.Add(new DataPoint(1, 0.78));
        ValLossPoints.Add(new DataPoint(2, 0.51));
        ValLossPoints.Add(new DataPoint(3, 0.39));
        ValLossPoints.Add(new DataPoint(4, 0.31));

        // Добавляем точки для графика точности
        TrainAccPoints.Add(new DataPoint(0, 72.15));
        TrainAccPoints.Add(new DataPoint(1, 88.42));
        TrainAccPoints.Add(new DataPoint(2, 94.10));
        TrainAccPoints.Add(new DataPoint(3, 97.21));
        TrainAccPoints.Add(new DataPoint(4, 98.42));

        ValAccPoints.Add(new DataPoint(0, 68.42));
        ValAccPoints.Add(new DataPoint(1, 87.12));
        ValAccPoints.Add(new DataPoint(2, 93.20));
        ValAccPoints.Add(new DataPoint(3, 96.45));
        ValAccPoints.Add(new DataPoint(4, 98.11));
    }

    private void InitializeConfusionMatrix()
    {
        ConfusionMatrix = new ConfusionMatrixViewModel(10);
    }
}
