using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class AnalysisViewModel : BaseViewModel
{
    private LayerBlockViewModel? _selectedLayer;
    private VisualizationMode _currentMode = VisualizationMode.Activations;
    private int _selectedFeatureMapIndex;
    private double _opacity = 0.7;
    private bool _showOverlay = true;
    private string? _originalImagePath;

    private string _imageDimensions = "28 x 28";
    private string _imageFormat = "PNG";
    private bool _hasVisualization;
    private string _visualizationImage;
    private string _currentModeIcon;

    private ObservableCollection<TopClassViewModel> _topClasses;
    private double _peakActivation = 0.856;
    private string _peakLocation = "(14, 12)";


    public AnalysisViewModel()
    {
        // Инициализация коллекций
        Layers = new ObservableCollection<LayerBlockViewModel>();
        FeatureMaps = new ObservableCollection<FeatureMapViewModel>();
        FilterVisualizations = new ObservableCollection<FilterViewModel>();
        AvailableImages = new ObservableCollection<string>();

        // Команды
        LoadImageCommand = new RelayCommand(LoadImage);
        AnalyzeCommand = new RelayCommand(Analyze, (_) => SelectedLayer != null);
        PreviousFeatureMapCommand = new RelayCommand(PreviousFeatureMap, (_) => SelectedFeatureMapIndex > 0);
        NextFeatureMapCommand = new RelayCommand(NextFeatureMap, (_) => SelectedFeatureMapIndex < FeatureMaps.Count - 1);
        ExportVisualizationCommand = new RelayCommand(ExportVisualization);

        // Загружаем демо-данные
        LoadDemoData();

        _currentModeIcon = "ActivationsIconGeometry";
        _visualizationImage = string.Empty;

        _topClasses = new ObservableCollection<TopClassViewModel>();
        LoadTopClasses();
    }

    // Коллекции
    public ObservableCollection<LayerBlockViewModel> Layers { get; }
    public ObservableCollection<FeatureMapViewModel> FeatureMaps { get; }
    public ObservableCollection<FilterViewModel> FilterVisualizations { get; }
    public ObservableCollection<string> AvailableImages { get; }

    // Свойства
    public LayerBlockViewModel? SelectedLayer
    {
        get => _selectedLayer;
        set
        {
            if (SetProperty(ref _selectedLayer, value))
            {
                OnPropertyChanged(nameof(CanAnalyze));
                LoadFeatureMaps();
            }
        }
    }

    public int SelectedFeatureMapIndex
    {
        get => _selectedFeatureMapIndex;
        set
        {
            if (SetProperty(ref _selectedFeatureMapIndex, value))
            {
                OnPropertyChanged(nameof(SelectedFeatureMap));
                OnPropertyChanged(nameof(FeatureMapDisplay));
            }
        }
    }

    public FeatureMapViewModel? SelectedFeatureMap =>
        FeatureMaps.Count > SelectedFeatureMapIndex ? FeatureMaps[SelectedFeatureMapIndex] : null;

    public string FeatureMapDisplay =>
        FeatureMaps.Count > 0 ? $"{SelectedFeatureMapIndex + 1}/{FeatureMaps.Count}" : "0/0";

    public VisualizationMode CurrentMode
    {
        get => _currentMode;
        set
        {
            if (SetProperty(ref _currentMode, value))
            {
                OnPropertyChanged(nameof(IsActivationsMode));
                OnPropertyChanged(nameof(IsFiltersMode));
                OnPropertyChanged(nameof(IsGradCamMode));
                LoadVisualization();
            }
        }
    }

    public bool IsActivationsMode => CurrentMode == VisualizationMode.Activations;
    public bool IsFiltersMode => CurrentMode == VisualizationMode.Filters;
    public bool IsGradCamMode => CurrentMode == VisualizationMode.GradCAM;

    public double Opacity
    {
        get => _opacity;
        set => SetProperty(ref _opacity, value);
    }

    public bool ShowOverlay
    {
        get => _showOverlay;
        set => SetProperty(ref _showOverlay, value);
    }

    public string? OriginalImagePath
    {
        get => _originalImagePath;
        set
        {
            if (SetProperty(ref _originalImagePath, value))
            {
                OnPropertyChanged(nameof(CanAnalyze));
            }
        }
    }

    public bool CanAnalyze => SelectedLayer != null && OriginalImagePath != null;

    // Команды
    public ICommand LoadImageCommand { get; }
    public ICommand AnalyzeCommand { get; }
    public ICommand PreviousFeatureMapCommand { get; }
    public ICommand NextFeatureMapCommand { get; }
    public ICommand ExportVisualizationCommand { get; }

    // Методы команд

    private void Analyze(object? parameter)
    {
        if (!CanAnalyze) return;
        LoadVisualization();
    }

    private void PreviousFeatureMap(object? parameter)
    {
        if (SelectedFeatureMapIndex > 0)
            SelectedFeatureMapIndex--;
    }

    private void NextFeatureMap(object? parameter)
    {
        if (SelectedFeatureMapIndex < FeatureMaps.Count - 1)
            SelectedFeatureMapIndex++;
    }

    private void ExportVisualization(object? parameter)
    {
        // Логика экспорта визуализации
    }

    // Вспомогательные методы
    private void LoadFeatureMaps()
    {
        FeatureMaps.Clear();
        if (SelectedLayer == null) return;

        // Демо-данные для feature maps
        var random = new System.Random(42);
        int mapCount = SelectedLayer.Type switch
        {
            "conv2d" => 6, // 6 filters
            "pooling" => 6,
            _ => 1
        };

        for (int i = 0; i < mapCount; i++)
        {
            FeatureMaps.Add(new FeatureMapViewModel
            {
                Index = i,
                Name = $"Feature Map {i + 1}",
                MinValue = random.NextDouble() * 0.5,
                MaxValue = 0.5 + random.NextDouble() * 0.5,
                MeanValue = 0.3 + random.NextDouble() * 0.4
            });
        }

        SelectedFeatureMapIndex = 0;
    }

    private void LoadVisualization()
    {
        HasVisualization = true;
        switch (CurrentMode)
        {
            case VisualizationMode.Activations:
                LoadActivations();
                break;
            case VisualizationMode.Filters:
                LoadFilters();
                break;
            case VisualizationMode.GradCAM:
                LoadGradCAM();
                break;
        }
        OnPropertyChanged(nameof(CurrentModeIcon));       
    }

    private void LoadActivations()
    {
        // Здесь будет логика загрузки активаций из сети
    }

    private void LoadFilters()
    {
        FilterVisualizations.Clear();
        if (SelectedLayer?.Type != "conv2d") return;

        // Демо-данные для фильтров
        var random = new System.Random(42);
        for (int i = 0; i < 6; i++)
        {
            FilterVisualizations.Add(new FilterViewModel
            {
                Index = i,
                Size = "5x5",
                MinWeight = -0.3,
                MaxWeight = 0.5,
                MeanWeight = 0.1
            });
        }
    }

    private void LoadGradCAM()
    {
        // Здесь будет логика Grad-CAM
    }

    private void LoadDemoData()
    {
        // Добавляем демо-слои
        Layers.Add(new LayerBlockViewModel
        {
            Id = "#1",
            Name = "Conv2D",
            Subtitle = "C1 • 6 filters",
            Type = "conv2d",
            Color = "#60a5fa"
        });

        Layers.Add(new LayerBlockViewModel
        {
            Id = "#2",
            Name = "Pooling",
            Subtitle = "S2 • AvgPool",
            Type = "pooling",
            Color = "#6ee7b7"
        });

        Layers.Add(new LayerBlockViewModel
        {
            Id = "#3",
            Name = "Conv2D",
            Subtitle = "C3 • 16 filters",
            Type = "conv2d",
            Color = "#60a5fa"
        });

        Layers.Add(new LayerBlockViewModel
        {
            Id = "#4",
            Name = "Activation",
            Subtitle = "ReLU",
            Type = "activation",
            Color = "#f0abfc"
        });

        Layers.Add(new LayerBlockViewModel
        {
            Id = "#5",
            Name = "Pooling",
            Subtitle = "S4 • AvgPool",
            Type = "pooling",
            Color = "#6ee7b7"
        });

        Layers.Add(new LayerBlockViewModel
        {
            Id = "#6",
            Name = "Flatten",
            Subtitle = "Vectorize",
            Type = "flatten",
            Color = "#fda4af"
        });

        SelectedLayer = Layers[0];
    }

    public string ImageDimensions
    {
        get => _imageDimensions;
        set => SetProperty(ref _imageDimensions, value);
    }

    public string ImageFormat
    {
        get => _imageFormat;
        set => SetProperty(ref _imageFormat, value);
    }

    public bool HasVisualization
    {
        get => _hasVisualization;
        set => SetProperty(ref _hasVisualization, value);
    }

    public string VisualizationImage
    {
        get => _visualizationImage;
        set => SetProperty(ref _visualizationImage, value);
    }

    public string CurrentModeIcon
    {
        get
        {
            return CurrentMode switch
            {
                VisualizationMode.Activations => "ActivationsIconGeometry",
                VisualizationMode.Filters => "FiltersIconGeometry",
                VisualizationMode.GradCAM => "GradCamIconGeometry",
                _ => "ActivationsIconGeometry"
            };
        }
    }

    // Обновляем LoadImage
    private void LoadImage(object? parameter)
    {
        // В реальном приложении - диалог выбора файла
        OriginalImagePath = "demo_image.png";
        AvailableImages.Add(OriginalImagePath);
        ImageDimensions = "28 x 28";
        ImageFormat = "PNG";
    }

    public ObservableCollection<TopClassViewModel> TopClasses
    {
        get => _topClasses;
        set => SetProperty(ref _topClasses, value);
    }

    public double PeakActivation
    {
        get => _peakActivation;
        set => SetProperty(ref _peakActivation, value);
    }

    public string PeakLocation
    {
        get => _peakLocation;
        set => SetProperty(ref _peakLocation, value);
    }

    private void LoadTopClasses()
    {
        _topClasses.Clear();
        _topClasses.Add(new TopClassViewModel { ClassIndex = "07", ClassName = "Digit 7", Probability = 0.92 });
        _topClasses.Add(new TopClassViewModel { ClassIndex = "01", ClassName = "Digit 1", Probability = 0.05 });
        _topClasses.Add(new TopClassViewModel { ClassIndex = "09", ClassName = "Digit 9", Probability = 0.02 });
        _topClasses.Add(new TopClassViewModel { ClassIndex = "04", ClassName = "Digit 4", Probability = 0.01 });
    }
}

