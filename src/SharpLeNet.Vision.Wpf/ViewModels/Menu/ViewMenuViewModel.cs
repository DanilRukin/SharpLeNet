using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class ViewMenuViewModel : BaseMenuViewModel
{
    private bool _isFullScreen;
    private bool _showLeftPanel = true;
    private bool _showRightPanel = true;
    private bool _showOutputConsole;
    private bool _showGrid = true;
    private bool _snapToGrid = true;
    private bool _showConnections = true;
    private bool _showCoordinates;
    private bool _showRulers;
    private string _currentTheme = "Dark";

    public ViewMenuViewModel()
    {
        // Команды
        ToggleFullScreenCommand = new RelayCommand(ToggleFullScreen);
        ZoomInCommand = new RelayCommand(ZoomIn);
        ZoomOutCommand = new RelayCommand(ZoomOut);
        ResetZoomCommand = new RelayCommand(ResetZoom);

        ToggleLeftPanelCommand = new RelayCommand(ToggleLeftPanel);
        ToggleRightPanelCommand = new RelayCommand(ToggleRightPanel);
        ToggleOutputConsoleCommand = new RelayCommand(ToggleOutputConsole);
        ToggleGridCommand = new RelayCommand(ToggleGrid);
        ToggleSnapToGridCommand = new RelayCommand(ToggleSnapToGrid);
        ToggleConnectionsCommand = new RelayCommand(ToggleConnections);
        ToggleCoordinatesCommand = new RelayCommand(ToggleCoordinates);
        ToggleRulersCommand = new RelayCommand(ToggleRulers);

        SelectDarkThemeCommand = new RelayCommand(() => SelectTheme("Dark"));
        SelectLightThemeCommand = new RelayCommand(() => SelectTheme("Light"));
        SelectHighContrastCommand = new RelayCommand(() => SelectTheme("HighContrast"));

        ResetAllCommand = new RelayCommand(ResetAll);
        ApplyCommand = new RelayCommand(Apply);
        CancelCommand = new RelayCommand(Cancel);
    }

    // Состояния
    public bool IsFullScreen
    {
        get => _isFullScreen;
        set => SetProperty(ref _isFullScreen, value);
    }

    public bool ShowLeftPanel
    {
        get => _showLeftPanel;
        set => SetProperty(ref _showLeftPanel, value);
    }

    public bool ShowRightPanel
    {
        get => _showRightPanel;
        set => SetProperty(ref _showRightPanel, value);
    }

    public bool ShowOutputConsole
    {
        get => _showOutputConsole;
        set => SetProperty(ref _showOutputConsole, value);
    }

    public bool ShowGrid
    {
        get => _showGrid;
        set => SetProperty(ref _showGrid, value);
    }

    public bool SnapToGrid
    {
        get => _snapToGrid;
        set => SetProperty(ref _snapToGrid, value);
    }

    public bool ShowConnections
    {
        get => _showConnections;
        set => SetProperty(ref _showConnections, value);
    }

    public bool ShowCoordinates
    {
        get => _showCoordinates;
        set => SetProperty(ref _showCoordinates, value);
    }

    public bool ShowRulers
    {
        get => _showRulers;
        set => SetProperty(ref _showRulers, value);
    }

    public string CurrentTheme
    {
        get => _currentTheme;
        set => SetProperty(ref _currentTheme, value);
    }

    // Команды
    public ICommand ToggleFullScreenCommand { get; }
    public ICommand ZoomInCommand { get; }
    public ICommand ZoomOutCommand { get; }
    public ICommand ResetZoomCommand { get; }

    public ICommand ToggleLeftPanelCommand { get; }
    public ICommand ToggleRightPanelCommand { get; }
    public ICommand ToggleOutputConsoleCommand { get; }
    public ICommand ToggleGridCommand { get; }
    public ICommand ToggleSnapToGridCommand { get; }
    public ICommand ToggleConnectionsCommand { get; }
    public ICommand ToggleCoordinatesCommand { get; }
    public ICommand ToggleRulersCommand { get; }

    public ICommand SelectDarkThemeCommand { get; }
    public ICommand SelectLightThemeCommand { get; }
    public ICommand SelectHighContrastCommand { get; }

    public ICommand ResetAllCommand { get; }
    public ICommand ApplyCommand { get; }
    public ICommand CancelCommand { get; }

    private void ToggleFullScreen(object? parameter) => IsFullScreen = !IsFullScreen;
    private void ZoomIn(object? parameter) { /* Логика */ }
    private void ZoomOut(object? parameter) { /* Логика */ }
    private void ResetZoom(object? parameter) { /* Логика */ }

    private void ToggleLeftPanel(object? parameter) => ShowLeftPanel = !ShowLeftPanel;
    private void ToggleRightPanel(object? parameter) => ShowRightPanel = !ShowRightPanel;
    private void ToggleOutputConsole(object? parameter) => ShowOutputConsole = !ShowOutputConsole;
    private void ToggleGrid(object? parameter) => ShowGrid = !ShowGrid;
    private void ToggleSnapToGrid(object? parameter) => SnapToGrid = !SnapToGrid;
    private void ToggleConnections(object? parameter) => ShowConnections = !ShowConnections;
    private void ToggleCoordinates(object? parameter) => ShowCoordinates = !ShowCoordinates;
    private void ToggleRulers(object? parameter) => ShowRulers = !ShowRulers;

    private void SelectTheme(string theme) => CurrentTheme = theme;
    private void ResetAll(object? parameter) { /* Логика */ }
    private void Apply(object? parameter) => CloseCommand.Execute(null);
    private void Cancel(object? parameter) => CloseCommand.Execute(null);
}
