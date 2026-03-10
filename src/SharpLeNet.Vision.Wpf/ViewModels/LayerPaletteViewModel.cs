using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class LayerPaletteViewModel : BaseViewModel
{
    private LayerPaletteItemViewModel? _selectedItem;
    private string _searchFilter = string.Empty;
    private bool _showAdvancedLayers;

    public LayerPaletteViewModel()
    {
        // Инициализация коллекций
        Items = new ObservableCollection<LayerPaletteItemViewModel>();
        Categories = new ObservableCollection<string> { "Все", "Свертка", "Пулинг", "Активация", "Линейные" };
        SelectedCategory = "Все";

        // Команды
        LayerSelectedCommand = new RelayCommand<LayerPaletteItemViewModel>(OnLayerSelected);
        ManageLayersCommand = new RelayCommand(OnManageLayers);
        FilterChangedCommand = new RelayCommand<string>(OnFilterChanged);
        ToggleAdvancedCommand = new RelayCommand(OnToggleAdvanced);

        // Загружаем слои
        LoadLayerItems();
    }

    public ObservableCollection<LayerPaletteItemViewModel> Items { get; }
    public ObservableCollection<string> Categories { get; }

    public string SelectedCategory { get; set; }

    public LayerPaletteItemViewModel? SelectedItem
    {
        get => _selectedItem;
        set => SetProperty(ref _selectedItem, value);
    }

    public string SearchFilter
    {
        get => _searchFilter;
        set
        {
            if (SetProperty(ref _searchFilter, value))
            {
                ApplyFilter();
            }
        }
    }

    public bool ShowAdvancedLayers
    {
        get => _showAdvancedLayers;
        set
        {
            if (SetProperty(ref _showAdvancedLayers, value))
            {
                LoadLayerItems();
            }
        }
    }

    // Команды
    public ICommand LayerSelectedCommand { get; }
    public ICommand ManageLayersCommand { get; }
    public ICommand FilterChangedCommand { get; }
    public ICommand ToggleAdvancedCommand { get; }

    private void LoadLayerItems()
    {
        Items.Clear();

        // Базовые слои
        Items.Add(new LayerPaletteItemViewModel
        {
            Name = "Conv2D",
            Subtitle = "Фильтры, шаг, padding",
            Badge = "3×3",
            Color = "#60a5fa",
            BorderColor = "#60a5fa59",
            BackgroundGradient = "LinearGradient 0,0,0,1 #0b1a2a #081624",
            Icon = "ConvIcon",
            LayerType = LayerType.Conv2D,
            Category = "Свертка",
            Description = "Сверточный слой для выделения признаков",
            DefaultParams = new[] { ("K", "3×3"), ("S", "1"), ("P", "0") }
        });

        Items.Add(new LayerPaletteItemViewModel
        {
            Name = "Pooling",
            Subtitle = "Downsample feature maps",
            Badge = "2×2",
            Color = "#6ee7b7",
            BorderColor = "#6ee7b759",
            BackgroundGradient = "LinearGradient 0,0,0,1 #071f16 #061a12",
            Icon = "PoolIcon",
            LayerType = LayerType.Pooling,
            Category = "Пулинг",
            Description = "Уменьшение размерности карт признаков",
            DefaultParams = new[] { ("K", "2×2"), ("S", "2") }
        });

        Items.Add(new LayerPaletteItemViewModel
        {
            Name = "Linear",
            Subtitle = "Полносвязный слой",
            Badge = "120",
            Color = "#fdba74",
            BorderColor = "#fdba7459",
            BackgroundGradient = "LinearGradient 0,0,0,1 #241407 #1f1005",
            Icon = "LinearIcon",
            LayerType = LayerType.Linear,
            Category = "Линейные",
            Description = "Полносвязное соединение нейронов",
            DefaultParams = new[] { ("Out", "120") }
        });

        Items.Add(new LayerPaletteItemViewModel
        {
            Name = "Activation",
            Subtitle = "Нелинейность",
            Badge = "ReLU",
            Color = "#f0abfc",
            BorderColor = "#f0abfc59",
            BackgroundGradient = "LinearGradient 0,0,0,1 #220a1a #1b0814",
            Icon = "ActivationIcon",
            LayerType = LayerType.Activation,
            Category = "Активация",
            Description = "Функция активации",
            DefaultParams = new[] { ("Type", "ReLU") }
        });

        Items.Add(new LayerPaletteItemViewModel
        {
            Name = "Flatten",
            Subtitle = "Сведение тензора",
            Badge = "N→1D",
            Color = "#fda4af",
            BorderColor = "#fda4af59",
            BackgroundGradient = "LinearGradient 0,0,0,1 #26090d #1f070a",
            Icon = "FlattenIcon",
            LayerType = LayerType.Flatten,
            Category = "Преобразование",
            Description = "Преобразование многомерного тензора в вектор",
            DefaultParams = new[] { ("Dim", "-1") }
        });

        // Дополнительные слои (если включены)
        if (ShowAdvancedLayers)
        {
            Items.Add(new LayerPaletteItemViewModel
            {
                Name = "Dropout",
                Subtitle = "Регуляризация",
                Badge = "0.5",
                Color = "#94a3b8",
                BorderColor = "#94a3b859",
                BackgroundGradient = "LinearGradient 0,0,0,1 #1e293b #0f172a",
                Icon = "DropoutIcon",
                LayerType = LayerType.Dropout,
                Category = "Регуляризация",
                Description = "Случайное отключение нейронов",
                DefaultParams = new[] { ("Rate", "0.5") }
            });

            Items.Add(new LayerPaletteItemViewModel
            {
                Name = "BatchNorm",
                Subtitle = "Нормализация",
                Badge = "2D",
                Color = "#c084fc",
                BorderColor = "#c084fc59",
                BackgroundGradient = "LinearGradient 0,0,0,1 #2e1065 #1e1b4b",
                Icon = "BatchNormIcon",
                LayerType = LayerType.BatchNorm,
                Category = "Нормализация",
                Description = "Пакетная нормализация",
                DefaultParams = new[] { ("Momentum", "0.99") }
            });
        }

        ApplyFilter();
    }

    private void ApplyFilter()
    {
        // В реальном приложении здесь будет фильтрация коллекции
        // Сейчас просто уведомляем об изменении
        OnPropertyChanged(nameof(Items));
    }

    private void OnLayerSelected(LayerPaletteItemViewModel? layer)
    {
        if (layer != null)
        {
            SelectedItem = layer;
            // Здесь можно добавить логику при выборе слоя
        }
    }

    private void OnManageLayers(object? parameter)
    {
        // Открыть диалог управления пользовательскими слоями
    }

    private void OnFilterChanged(string? category)
    {
        if (!string.IsNullOrEmpty(category))
        {
            SelectedCategory = category;
            ApplyFilter();
        }
    }

    private void OnToggleAdvanced(object? parameter)
    {
        ShowAdvancedLayers = !ShowAdvancedLayers;
    }
}
