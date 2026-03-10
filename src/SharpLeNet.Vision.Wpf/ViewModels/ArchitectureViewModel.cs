using SharpLeNet.Core;
using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class ArchitectureViewModel : BaseViewModel
{
    private readonly Model _model;
    private Point _mousePosition;

    public ArchitectureViewModel(Model model)
    {
        _model = model;
        LayerBlocks = new ObservableCollection<LayerBlockViewModel>();
        Connections = new ObservableCollection<ConnectionViewModel>();

        // Команды
        AddLayerCommand = new RelayCommand<LayerPaletteItemViewModel>(AddLayer);
        RemoveLayerCommand = new RelayCommand<LayerBlockViewModel>(RemoveLayer);
        ConnectLayersCommand = new RelayCommand<Tuple<LayerBlockViewModel, LayerBlockViewModel>>(ConnectLayers);
        CanvasClickCommand = new RelayCommand<Point>(OnCanvasClick);
        BlockMovedCommand = new RelayCommand<Tuple<LayerBlockViewModel, Point>>(OnBlockMoved);

        // Загружаем демо-блоки для прототипа
        LoadDemoArchitecture();
    }

    public ObservableCollection<LayerBlockViewModel> LayerBlocks { get; }
    public ObservableCollection<ConnectionViewModel> Connections { get; }

    public Point MousePosition
    {
        get => _mousePosition;
        set => SetProperty(ref _mousePosition, value);
    }

    public ICommand AddLayerCommand { get; }
    public ICommand RemoveLayerCommand { get; }
    public ICommand ConnectLayersCommand { get; }
    public ICommand CanvasClickCommand { get; }
    public ICommand BlockMovedCommand { get; }

    private void LoadDemoArchitecture()
    {
        // Input block
        LayerBlocks.Add(new LayerBlockViewModel
        {
            Id = "#0",
            Name = "Input",
            Subtitle = "MNIST",
            Type = "input",
            Color = "#ffffff",
            BorderColor = "#ffffff1A",
            BackgroundGradient = "LinearGradient 0,0,0,1 #242427 #202022",
            OutputDim = "28×28×1",
            Position = new Point(40, 90),
            Icon = "ImageIcon"
        });

        // Conv1
        LayerBlocks.Add(new LayerBlockViewModel
        {
            Id = "#1",
            Name = "Conv2D",
            Subtitle = "C1 • 6 filters",
            Type = "conv2d",
            Color = "#60a5fa",
            BorderColor = "#60a5fa59",
            BackgroundGradient = "LinearGradient 0,0,0,1 #0b1a2a #081624",
            OutputDim = "24×24×6",
            Position = new Point(290, 86),
            Icon = "ScanIcon",
            Parameters = new ObservableCollection<ParameterViewModel>
                {
                    new() { Label = "K", Value = "5×5" },
                    new() { Label = "S", Value = "1" },
                    new() { Label = "P", Value = "0" }
                }
        });

        // Соединение между Input и Conv1
        Connections.Add(new ConnectionViewModel
        {
            SourceId = "#0",
            TargetId = "#1",
            SourceBlock = LayerBlocks[0],
            TargetBlock = LayerBlocks[1]
        });

        // Добавить остальные блоки...
    }

    public void LoadLeNetTemplate()
    {
        LayerBlocks.Clear();
        Connections.Clear();
        LoadDemoArchitecture();
    }

    private void AddLayer(LayerPaletteItemViewModel? paletteItem)
    {
        if (paletteItem == null) return;

        var newLayer = new LayerBlockViewModel
        {
            Id = $"#{LayerBlocks.Count}",
            Name = paletteItem.Name,
            Subtitle = paletteItem.Subtitle,
            Type = paletteItem.LayerType.ToString().ToLower(),
            Color = paletteItem.Color,
            BorderColor = paletteItem.BorderColor,
            BackgroundGradient = paletteItem.BackgroundGradient,
            Icon = paletteItem.Icon,
            Position = new Point(MousePosition.X - 105, MousePosition.Y - 60) // Center on mouse
        };

        LayerBlocks.Add(newLayer);
    }

    private void RemoveLayer(LayerBlockViewModel? layer)
    {
        if (layer == null) return;

        // Remove all connections to/from this layer
        for (int i = Connections.Count - 1; i >= 0; i--)
        {
            if (Connections[i].SourceId == layer.Id || Connections[i].TargetId == layer.Id)
                Connections.RemoveAt(i);
        }

        LayerBlocks.Remove(layer);
    }

    private void ConnectLayers(Tuple<LayerBlockViewModel, LayerBlockViewModel>? connection)
    {
        if (connection == null) return;

        var (source, target) = connection;

        // Проверка совместимости размерностей
        if (IsCompatible(source, target))
        {
            Connections.Add(new ConnectionViewModel
            {
                SourceId = source.Id,
                TargetId = target.Id,
                SourceBlock = source,
                TargetBlock = target
            });
        }
    }

    private bool IsCompatible(LayerBlockViewModel source, LayerBlockViewModel target)
    {
        // Здесь будет логика проверки совместимости размерностей
        return true; // Для прототипа
    }

    private void OnCanvasClick(Point point)
    {
        MousePosition = point;
    }

    private void OnBlockMoved(Tuple<LayerBlockViewModel, Point>? moveInfo)
    {
        if (moveInfo == null) return;

        var (block, newPosition) = moveInfo;
        block.Position = newPosition;

        // Обновить все соединения, связанные с этим блоком
        foreach (var conn in Connections)
        {
            if (conn.SourceId == block.Id)
                conn.UpdatePath();
            else if (conn.TargetId == block.Id)
                conn.UpdatePath();
        }
    }
}
