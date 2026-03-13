using SharpLeNet.Core;
using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class MainViewModel : BaseViewModel
{
    private readonly Model _model;
    private bool _isTraining;
    private string _projectName;
    private string _currentMode;
    private BaseViewModel? _selectedTab;
    private RibbonViewModel? _ribbon;

    public MainViewModel()
    {
        _model = new Model();
        _projectName = "LeNet • MNIST";
        _currentMode = "Architect";

        // Инициализация дочерних ViewModel
        _ribbon = new RibbonViewModel(this);
        LayerPalette = new LayerPaletteViewModel();
        Architecture = new ArchitectureViewModel(_model);
        Properties = new PropertiesViewModel();
        Training = new TrainingViewModel();
        StatusBar = new StatusBarViewModel();
        Tabs = new ObservableCollection<TabViewModel>
            {
                new TabViewModel("Architect", typeof(ArchitectureViewModel), true),
                new TabViewModel("Training", typeof(TrainingViewModel), false),
                new TabViewModel("Analysis", typeof(AnalysisViewModel), false),
                new TabViewModel("Playground", typeof(PlaygroundViewModel), false)
            };

        SelectedTab = Training;

        // Команды
        StartTrainingCommand = new RelayCommand(StartTraining, (_) => !IsTraining);
        PauseTrainingCommand = new RelayCommand(PauseTraining, (_) => IsTraining);
        StopTrainingCommand = new RelayCommand(StopTraining, (_) => IsTraining);
        QuickStartCommand = new RelayCommand(QuickStart);
        ToggleSidebarCommand = new RelayCommand(ToggleSidebar);
        NavigateToTabCommand = new RelayCommand<TabViewModel>(NavigateToTab);
    }

    // Properties
    public RibbonViewModel Ribbon => _ribbon ??= new RibbonViewModel(this);
    public LayerPaletteViewModel LayerPalette { get; }
    public ArchitectureViewModel Architecture { get; }
    public PropertiesViewModel Properties { get; }
    public TrainingViewModel Training { get; }
    public StatusBarViewModel StatusBar { get; }
    public ObservableCollection<TabViewModel> Tabs { get; }

    public string ProjectName
    {
        get => _projectName;
        set => SetProperty(ref _projectName, value);
    }

    public string CurrentMode
    {
        get => _currentMode;
        set => SetProperty(ref _currentMode, value);
    }

    public bool IsTraining
    {
        get => _isTraining;
        private set => SetProperty(ref _isTraining, value);
    }

    public BaseViewModel? SelectedTab
    {
        get => _selectedTab;
        set
        {
            if (SetProperty(ref _selectedTab, value) && value != null)
            {
                CurrentMode = value.GetType().Name.Replace("ViewModel", "");
            }
        }
    }

    // Commands
    public ICommand StartTrainingCommand { get; }
    public ICommand PauseTrainingCommand { get; }
    public ICommand StopTrainingCommand { get; }
    public ICommand QuickStartCommand { get; }
    public ICommand ToggleSidebarCommand { get; }
    public ICommand NavigateToTabCommand { get; }

    private void NavigateToTab(TabViewModel? tab)
    {
        if (tab == null) return;

        // Снимаем активность со всех табов
        foreach (var t in Tabs)
        {
            t.IsActive = false;
        }

        // Активируем выбранный таб
        tab.IsActive = true;

        // Устанавливаем соответствующий ViewModel
        if (tab.ViewModelType == typeof(ArchitectureViewModel))
            SelectedTab = Architecture;
        else if (tab.ViewModelType == typeof(TrainingViewModel))
            SelectedTab = Training;
        //else if (tab.ViewModelType == typeof(AnalysisViewModel))
        //    SelectedTab = Analysis;
        //else if (tab.ViewModelType == typeof(PlaygroundViewModel))
        //    SelectedTab = Playground;
    }

    private void StartTraining(object? parameter)
    {
        IsTraining = true;
        Training.StartCommand.Execute(parameter);
        StatusBar.UpdateTrainingStatus(Training.CurrentEpoch, Training.CurrentBatch, Training.CurrentLoss, Training.LearningRate, Training.Eta);
    }

    private void PauseTraining(object? parameter)
    {
        Training.PauseCommand.Execute(parameter);
        StatusBar.UpdateTrainingStatus(Training.CurrentEpoch, Training.CurrentBatch, Training.CurrentLoss, Training.LearningRate, Training.Eta);
    }

    private void StopTraining(object? parameter)
    {
        IsTraining = false;
        Training.StopCommand.Execute(parameter);
        StatusBar.UpdateTrainingStatus(Training.CurrentEpoch, Training.CurrentBatch, Training.CurrentLoss, Training.LearningRate, Training.Eta);
    }

    private void QuickStart(object? parameter)
    {
        // Load pre-defined LeNet architecture
        Architecture.LoadLeNetTemplate();
        SelectedTab = Architecture;
    }

    private void ToggleSidebar(object? parameter)
    {
        // Implement sidebar toggle logic
    }
}
