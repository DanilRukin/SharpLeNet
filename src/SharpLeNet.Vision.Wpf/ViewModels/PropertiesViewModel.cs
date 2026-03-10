using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class PropertiesViewModel : BaseViewModel
{
    private LayerBlockViewModel? _selectedLayer;
    private string _selectedActivation = "ReLU";
    private int _kernelSize = 5;
    private int _stride = 1;
    private int _padding = 0;
    private int _filters = 6;
    private int _outputUnits = 120;
    private double _dropoutRate = 0.5;
    private bool _useBias = true;
    private string _paddingMode = "Same";
    private bool _isDirty;

    public PropertiesViewModel()
    {
        // Доступные функции активации
        ActivationFunctions = new ObservableCollection<string>
            {
                "ReLU",
                "LeakyReLU",
                "Tanh",
                "Sigmoid",
                "GELU",
                "Swish"
            };

        PaddingModes = new ObservableCollection<string>
            {
                "Same",
                "Valid",
                "Custom"
            };

        // Команды
        ApplyCommand = new RelayCommand(ApplyChanges, () => IsDirty);
        ResetCommand = new RelayCommand(ResetChanges, () => IsDirty);
        RefreshStatsCommand = new RelayCommand(RefreshStats);
    }

    // Свойства для разных типов слоев
    public LayerBlockViewModel? SelectedLayer
    {
        get => _selectedLayer;
        set
        {
            if (SetProperty(ref _selectedLayer, value))
            {
                LoadLayerProperties();
                OnPropertyChanged(nameof(LayerType));
                OnPropertyChanged(nameof(IsConvLayer));
                OnPropertyChanged(nameof(IsPoolingLayer));
                OnPropertyChanged(nameof(IsLinearLayer));
                OnPropertyChanged(nameof(IsActivationLayer));
                OnPropertyChanged(nameof(IsDropoutLayer));
            }
        }
    }

    public string LayerType => SelectedLayer?.Type ?? "None";
    public bool IsConvLayer => LayerType == "conv2d";
    public bool IsPoolingLayer => LayerType == "pooling";
    public bool IsLinearLayer => LayerType == "linear";
    public bool IsActivationLayer => LayerType == "activation";
    public bool IsDropoutLayer => LayerType == "dropout";

    // Общие свойства
    public ObservableCollection<string> ActivationFunctions { get; }
    public ObservableCollection<string> PaddingModes { get; }

    public string SelectedActivation
    {
        get => _selectedActivation;
        set => SetProperty(ref _selectedActivation, value);
    }

    // Conv2D свойства
    public int KernelSize
    {
        get => _kernelSize;
        set
        {
            if (SetProperty(ref _kernelSize, value))
            {
                IsDirty = true;
                RecalculateOutputDimensions();
            }
        }
    }

    public int Stride
    {
        get => _stride;
        set
        {
            if (SetProperty(ref _stride, value))
            {
                IsDirty = true;
                RecalculateOutputDimensions();
            }
        }
    }

    public int Padding
    {
        get => _padding;
        set
        {
            if (SetProperty(ref _padding, value))
            {
                IsDirty = true;
                RecalculateOutputDimensions();
            }
        }
    }

    public int Filters
    {
        get => _filters;
        set
        {
            if (SetProperty(ref _filters, value))
            {
                IsDirty = true;
                RecalculateOutputDimensions();
            }
        }
    }

    public string PaddingMode
    {
        get => _paddingMode;
        set
        {
            if (SetProperty(ref _paddingMode, value))
            {
                IsDirty = true;
                UpdatePaddingFromMode();
            }
        }
    }

    // Linear layer свойства
    public int OutputUnits
    {
        get => _outputUnits;
        set
        {
            if (SetProperty(ref _outputUnits, value))
            {
                IsDirty = true;
                RecalculateOutputDimensions();
            }
        }
    }

    public bool UseBias
    {
        get => _useBias;
        set
        {
            if (SetProperty(ref _useBias, value))
            {
                IsDirty = true;
            }
        }
    }

    // Dropout свойства
    public double DropoutRate
    {
        get => _dropoutRate;
        set
        {
            if (SetProperty(ref _dropoutRate, value))
            {
                IsDirty = true;
                OnPropertyChanged(nameof(DropoutRateDisplay));
            }
        }
    }

    public string DropoutRateDisplay => $"{DropoutRate:P0}";

    // Статистика
    private int _parameterCount;
    private double _flops;
    private string _outputShape = "—";
    private int _memoryUsage;
    private double _inferenceTime;
    private ObservableCollection<double> _weightDistribution;

    public int ParameterCount
    {
        get => _parameterCount;
        private set => SetProperty(ref _parameterCount, value);
    }

    public double Flops
    {
        get => _flops;
        private set => SetProperty(ref _flops, value);
    }

    public string FlopsDisplay => Flops switch
    {
        < 1e3 => $"{Flops:F0} FLOPs",
        < 1e6 => $"{Flops / 1e3:F2} KFLOPs",
        < 1e9 => $"{Flops / 1e6:F2} MFLOPs",
        _ => $"{Flops / 1e9:F2} GFLOPs"
    };

    public string OutputShape
    {
        get => _outputShape;
        private set => SetProperty(ref _outputShape, value);
    }

    public int MemoryUsage
    {
        get => _memoryUsage;
        private set => SetProperty(ref _memoryUsage, value);
    }

    public string MemoryDisplay => MemoryUsage switch
    {
        < 1024 => $"{MemoryUsage} B",
        < 1024 * 1024 => $"{MemoryUsage / 1024:F1} KB",
        _ => $"{MemoryUsage / (1024 * 1024):F1} MB"
    };

    public double InferenceTime
    {
        get => _inferenceTime;
        private set => SetProperty(ref _inferenceTime, value);
    }

    public string InferenceDisplay => $"{InferenceTime:F2} ms";

    public ObservableCollection<double> WeightDistribution
    {
        get => _weightDistribution ??= new ObservableCollection<double>();
        private set => SetProperty(ref _weightDistribution, value);
    }

    // Состояние
    public bool IsDirty
    {
        get => _isDirty;
        private set
        {
            if (SetProperty(ref _isDirty, value))
            {
                ApplyCommand.RaiseCanExecuteChanged();
                ResetCommand.RaiseCanExecuteChanged();
            }
        }
    }

    // Команды
    public RelayCommand ApplyCommand { get; }
    public RelayCommand ResetCommand { get; }
    public RelayCommand RefreshStatsCommand { get; }

    private void LoadLayerProperties()
    {
        if (SelectedLayer == null) return;

        // Сбрасываем флаг изменения
        IsDirty = false;

        // Загружаем свойства в зависимости от типа слоя
        switch (SelectedLayer.Type)
        {
            case "conv2d":
                KernelSize = 5;
                Stride = 1;
                Padding = 0;
                Filters = 6;
                SelectedActivation = "ReLU";
                break;
            case "pooling":
                KernelSize = 2;
                Stride = 2;
                Padding = 0;
                break;
            case "linear":
                OutputUnits = 120;
                UseBias = true;
                SelectedActivation = "ReLU";
                break;
            case "activation":
                SelectedActivation = SelectedLayer.Subtitle ?? "ReLU";
                break;
            case "dropout":
                DropoutRate = 0.5;
                break;
        }

        // Обновляем статистику
        RefreshStats();
    }

    private void RecalculateOutputDimensions()
    {
        if (SelectedLayer == null) return;

        // Здесь будет логика пересчета размерностей
        // Пока генерируем примерные значения
        switch (SelectedLayer.Type)
        {
            case "conv2d":
                var inputSize = 28; // Пример для MNIST
                var outputSize = ((inputSize - KernelSize + 2 * Padding) / Stride) + 1;
                OutputShape = $"{outputSize}×{outputSize}×{Filters}";
                ParameterCount = KernelSize * KernelSize * Filters * 3 + (UseBias ? Filters : 0); // Примерно
                Flops = ParameterCount * outputSize * outputSize * 2 / 1e6;
                MemoryUsage = outputSize * outputSize * Filters * 4; // 4 байта на float
                break;
            case "linear":
                OutputShape = $"{OutputUnits}";
                ParameterCount = 400 * OutputUnits + (UseBias ? OutputUnits : 0); // 400 - пример входа
                Flops = ParameterCount * 2 / 1e3;
                MemoryUsage = OutputUnits * 4;
                break;
        }
    }

    private void UpdatePaddingFromMode()
    {
        // Обновляем значение Padding в зависимости от выбранного режима
        Padding = PaddingMode switch
        {
            "Same" => KernelSize / 2,
            "Valid" => 0,
            _ => Padding
        };
    }

    private void RefreshStats()
    {
        // Обновляем статистику
        RecalculateOutputDimensions();

        // Генерируем примерное распределение весов
        WeightDistribution.Clear();
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            WeightDistribution.Add(random.NextDouble() * 0.8 - 0.3);
        }
    }

    private void ApplyChanges()
    {
        if (SelectedLayer == null) return;

        // Применяем изменения к слою
        SelectedLayer.Parameters.Clear();

        switch (SelectedLayer.Type)
        {
            case "conv2d":
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "K", Value = $"{KernelSize}×{KernelSize}" });
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "S", Value = Stride.ToString() });
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "P", Value = Padding.ToString() });
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "F", Value = Filters.ToString() });
                SelectedLayer.Subtitle = $"{Filters} filters";
                break;
            case "pooling":
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "K", Value = $"{KernelSize}×{KernelSize}" });
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "S", Value = Stride.ToString() });
                break;
            case "linear":
                SelectedLayer.Parameters.Add(new ParameterViewModel { Label = "Out", Value = OutputUnits.ToString() });
                SelectedLayer.Subtitle = $"{OutputUnits} units";
                break;
            case "activation":
                SelectedLayer.Subtitle = SelectedActivation;
                break;
        }

        SelectedLayer.OutputDim = OutputShape;
        IsDirty = false;
    }

    private void ResetChanges()
    {
        LoadLayerProperties();
    }
}
