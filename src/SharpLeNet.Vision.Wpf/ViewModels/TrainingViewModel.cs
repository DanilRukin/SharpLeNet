using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Timers;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class TrainingViewModel : BaseViewModel
{
    private readonly System.Timers.Timer _updateTimer;
    private bool _isTraining;
    private bool _isPaused;
    private int _currentEpoch;
    private int _totalEpochs = 15;
    private int _currentBatch;
    private int _totalBatches = 469;
    private double _currentLoss = 0.284;
    private double _bestLoss = 0.251;
    private double _currentAccuracy = 98.1;
    private double _bestAccuracy = 98.4;
    private double _learningRate = 0.001;
    private TimeSpan _eta = TimeSpan.FromMinutes(12).Add(TimeSpan.FromSeconds(48));
    private string _selectedMetric = "Loss";
    private ConfusionMatrixViewModel _confusionMatrix;

    public TrainingViewModel()
    {
        // Инициализация коллекций
        LossHistory = new ObservableCollection<DataPoint>();
        AccuracyHistory = new ObservableCollection<DataPoint>();
        AvailableMetrics = new ObservableCollection<string> { "Loss", "Accuracy", "LR" };

        // Инициализация confusion matrix
        _confusionMatrix = new ConfusionMatrixViewModel(10); // 10 классов для MNIST

        // Команды
        StartCommand = new RelayCommand(Start, (_) => !IsTraining && !IsPaused);
        PauseCommand = new RelayCommand(Pause, (_) => IsTraining && !IsPaused);
        ResumeCommand = new RelayCommand(Resume, (_) => IsPaused);
        StopCommand = new RelayCommand(Stop, (_) => IsTraining || IsPaused);
        ExportMetricsCommand = new RelayCommand(ExportMetrics);
        ClearHistoryCommand = new RelayCommand(ClearHistory);

        // Таймер для симуляции обновлений (в реальном приложении будет подключен к Trainer)
        _updateTimer = new System.Timers.Timer(1000);
        _updateTimer.Elapsed += OnUpdateTimerElapsed;

        // Загружаем демо-данные
        LoadDemoData();
    }

    // Коллекции
    public ObservableCollection<DataPoint> LossHistory { get; }
    public ObservableCollection<DataPoint> AccuracyHistory { get; }
    public ObservableCollection<string> AvailableMetrics { get; }

    public ConfusionMatrixViewModel ConfusionMatrix
    {
        get => _confusionMatrix;
        set => SetProperty(ref _confusionMatrix, value);
    }

    // Свойства состояния
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

    public int CurrentEpoch
    {
        get => _currentEpoch;
        set
        {
            if (SetProperty(ref _currentEpoch, value))
            {
                OnPropertyChanged(nameof(EpochProgress));
                OnPropertyChanged(nameof(EpochDisplay));
            }
        }
    }

    public int TotalEpochs
    {
        get => _totalEpochs;
        set
        {
            if (SetProperty(ref _totalEpochs, value))
            {
                OnPropertyChanged(nameof(EpochProgress));
                OnPropertyChanged(nameof(EpochDisplay));
            }
        }
    }

    public double EpochProgress => (double)CurrentEpoch / TotalEpochs * 100;
    public string EpochDisplay => $"{CurrentEpoch:D2}/{TotalEpochs:D2}";

    public int CurrentBatch
    {
        get => _currentBatch;
        set
        {
            if (SetProperty(ref _currentBatch, value))
            {
                OnPropertyChanged(nameof(BatchProgress));
                OnPropertyChanged(nameof(BatchDisplay));
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
                OnPropertyChanged(nameof(BatchProgress));
                OnPropertyChanged(nameof(BatchDisplay));
            }
        }
    }

    public double BatchProgress => (double)CurrentBatch / TotalBatches * 100;
    public string BatchDisplay => $"{CurrentBatch}/{TotalBatches}";

    public double CurrentLoss
    {
        get => _currentLoss;
        set
        {
            if (SetProperty(ref _currentLoss, value))
            {
                OnPropertyChanged(nameof(LossDisplay));
            }
        }
    }

    public string LossDisplay => _currentLoss.ToString("F3");

    public double BestLoss
    {
        get => _bestLoss;
        set => SetProperty(ref _bestLoss, value);
    }

    public double CurrentAccuracy
    {
        get => _currentAccuracy;
        set
        {
            if (SetProperty(ref _currentAccuracy, value))
            {
                OnPropertyChanged(nameof(AccuracyDisplay));
            }
        }
    }

    public string AccuracyDisplay => _currentAccuracy.ToString("F1") + "%";

    public double BestAccuracy
    {
        get => _bestAccuracy;
        set => SetProperty(ref _bestAccuracy, value);
    }

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

    public string LearningRateDisplay => _learningRate.ToString("E2");

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

    public string SelectedMetric
    {
        get => _selectedMetric;
        set
        {
            if (SetProperty(ref _selectedMetric, value))
            {
                OnPropertyChanged(nameof(SelectedMetricData));
            }
        }
    }

    public ObservableCollection<DataPoint> SelectedMetricData
    {
        get
        {
            return SelectedMetric switch
            {
                "Loss" => LossHistory,
                "Accuracy" => AccuracyHistory,
                _ => LossHistory
            };
        }
    }

    // Команды
    public ICommand StartCommand { get; }
    public ICommand PauseCommand { get; }
    public ICommand ResumeCommand { get; }
    public ICommand StopCommand { get; }
    public ICommand ExportMetricsCommand { get; }
    public ICommand ClearHistoryCommand { get; }

    // Методы команд
    private void Start(object? parameter)
    {
        IsTraining = true;
        IsPaused = false;
        _updateTimer.Start();
    }

    private void Pause(object? parameter)
    {
        IsPaused = true;
        _updateTimer.Stop();
    }

    private void Resume(object? parameter)
    {
        IsPaused = false;
        _updateTimer.Start();
    }

    private void Stop(object? parameter)
    {
        IsTraining = false;
        IsPaused = false;
        _updateTimer.Stop();
        CurrentEpoch = 0;
        CurrentBatch = 0;
    }

    private void ExportMetrics(object? parameter)
    {
        // Логика экспорта метрик в CSV/JSON
    }

    private void ClearHistory(object? parameter)
    {
        LossHistory.Clear();
        AccuracyHistory.Clear();
    }

    // Публичные методы для управления извне
    public void StartTraining()
    {
        Start(null);
    }

    public void PauseTraining()
    {
        Pause(null);
    }

    public void StopTraining()
    {
        Stop(null);
    }

    public void UpdateMetrics(double loss, double accuracy, int epoch, int batch)
    {
        CurrentLoss = loss;
        CurrentAccuracy = accuracy;
        CurrentEpoch = epoch;
        CurrentBatch = batch;

        LossHistory.Add(new DataPoint(epoch + (double)batch / TotalBatches, loss));
        AccuracyHistory.Add(new DataPoint(epoch + (double)batch / TotalBatches, accuracy));

        // Обновляем ETA
        if (epoch > 0 || batch > 0)
        {
            var progress = (epoch * TotalBatches + batch) / (double)(TotalEpochs * TotalBatches);
            if (progress > 0)
            {
                var elapsed = TimeSpan.FromSeconds(DateTime.Now.TimeOfDay.TotalSeconds); // В реальном приложении нужно хранить время старта
                Eta = TimeSpan.FromSeconds(elapsed.TotalSeconds / progress - elapsed.TotalSeconds);
            }
        }
    }

    private void OnUpdateTimerElapsed(object? sender, ElapsedEventArgs e)
    {
        // Симуляция обновлений для демо
        if (!IsTraining || IsPaused) return;

        // Генерируем случайные данные для демонстрации
        var random = new Random();
        var newLoss = Math.Max(0.05, CurrentLoss * 0.99 + random.NextDouble() * 0.01 - 0.005);
        var newAccuracy = Math.Min(99.5, CurrentAccuracy * 1.001 + random.NextDouble() * 0.1 - 0.05);

        CurrentBatch++;
        if (CurrentBatch > TotalBatches)
        {
            CurrentBatch = 1;
            CurrentEpoch++;
        }

        System.Windows.Application.Current.Dispatcher.Invoke(() =>
        {
            UpdateMetrics(newLoss, newAccuracy, CurrentEpoch, CurrentBatch);
        });
    }

    private void LoadDemoData()
    {
        // Загружаем демо-данные для графиков
        for (int i = 0; i < 50; i++)
        {
            var x = i * 0.1;
            LossHistory.Add(new DataPoint(x, 1.0 / (1 + x) + Math.Sin(x) * 0.1));
            AccuracyHistory.Add(new DataPoint(x, 100 - 50 / (1 + x) + Math.Cos(x) * 2));
        }
    }
}
