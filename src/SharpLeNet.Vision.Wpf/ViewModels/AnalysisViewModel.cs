using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class AnalysisViewModel : BaseViewModel
{
    private LayerBlockViewModel? _selectedLayer;
    private int _selectedFeatureMapIndex;
    private VisualizationMode _currentMode = VisualizationMode.Activations;
    private double _opacity = 0.7;
    private bool _showOverlay = true;
    private string? _originalImagePath;

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
    }

    public ObservableCollection<LayerBlockViewModel> Layers { get; }
    public ObservableCollection<FeatureMapViewModel> FeatureMaps { get; }
    public ObservableCollection<FilterViewModel> FilterVisualizations { get; }
    public ObservableCollection<string> AvailableImages { get; }

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

    private void LoadImage(object? parameter)
    {
        // В реальном приложении - диалог выбора файла
        OriginalImagePath = "demo_image.png";
        AvailableImages.Add(OriginalImagePath);
    }

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

    private void LoadFeatureMaps()
    {
        FeatureMaps.Clear();
        if (SelectedLayer == null) return;

        // Демо-данные для feature maps
        var random = new Random(42);
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
                MeanValue = 0.3 + random.NextDouble() * 0.4,
                HeatmapData = GenerateHeatmapData(random)
            });
        }

        SelectedFeatureMapIndex = 0;
    }

    private void LoadVisualization()
    {
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
        var random = new Random(42);
        for (int i = 0; i < 6; i++)
        {
            FilterVisualizations.Add(new FilterViewModel
            {
                Index = i,
                Size = "5x5",
                Weights = GenerateWeights(random),
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

    private double[,] GenerateHeatmapData(Random random)
    {
        var data = new double[28, 28];
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                data[y, x] = Math.Exp(-((x - 14) * (x - 14) + (y - 14) * (y - 14)) / 100) * random.NextDouble();
        return data;
    }

    private double[,] GenerateWeights(Random random)
    {
        var weights = new double[5, 5];
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                weights[y, x] = random.NextDouble() * 0.8 - 0.3;
        return weights;
    }

    private void LoadDemoData()
    {
        // Добавляем демо-слои
        var convLayer = new LayerBlockViewModel
        {
            Id = "#1",
            Name = "Conv2D",
            Subtitle = "C1 • 6 filters",
            Type = "conv2d",
            Color = "#60a5fa"
        };
        Layers.Add(convLayer);

        var poolLayer = new LayerBlockViewModel
        {
            Id = "#2",
            Name = "Pooling",
            Subtitle = "S2 • AvgPool",
            Type = "pooling",
            Color = "#6ee7b7"
        };
        Layers.Add(poolLayer);

        SelectedLayer = convLayer;
    }
}

