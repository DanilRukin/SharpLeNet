using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class HomeMenuViewModel : BaseMenuViewModel
{
    private int _totalProjects = 12;
    private double _bestAccuracy = 96.8;
    private string _totalTrainingTime = "24ч 45мин";
    private string _lastUpdate = "Сегодня в 14:30";

    public HomeMenuViewModel()
    {
        RecentProjects = new ObservableCollection<RecentProjectViewModel>();
        LoadRecentProjects();

        // Команды
        NewProjectCommand = new RelayCommand(NewProject);
        OpenProjectCommand = new RelayCommand(OpenProject);
        OpenExamplesCommand = new RelayCommand(OpenExamples);
        OpenDatasetsCommand = new RelayCommand(OpenDatasets);
        ShowTutorialsCommand = new RelayCommand(ShowTutorials);
        OpenDocumentationCommand = new RelayCommand(OpenDocumentation);
        OpenSupportCommand = new RelayCommand(OpenSupport);
    }

    public ObservableCollection<RecentProjectViewModel> RecentProjects { get; }

    public int TotalProjects
    {
        get => _totalProjects;
        set => SetProperty(ref _totalProjects, value);
    }

    public double BestAccuracy
    {
        get => _bestAccuracy;
        set => SetProperty(ref _bestAccuracy, value);
    }

    public string TotalTrainingTime
    {
        get => _totalTrainingTime;
        set => SetProperty(ref _totalTrainingTime, value);
    }

    public string LastUpdate
    {
        get => _lastUpdate;
        set => SetProperty(ref _lastUpdate, value);
    }

    // Команды
    public ICommand NewProjectCommand { get; }
    public ICommand OpenProjectCommand { get; }
    public ICommand OpenExamplesCommand { get; }
    public ICommand OpenDatasetsCommand { get; }
    public ICommand ShowTutorialsCommand { get; }
    public ICommand OpenDocumentationCommand { get; }
    public ICommand OpenSupportCommand { get; }

    private void LoadRecentProjects()
    {
        RecentProjects.Add(new RecentProjectViewModel
        {
            Name = "mnist_experiment_v3.sharpnet",
            Icon = "ScanEye",
            Color = "Gold",
            TimeAgo = "2 часа назад",
            Status = "Trained (98.1%)",
            StatusColor = "Emerald"
        });

        RecentProjects.Add(new RecentProjectViewModel
        {
            Name = "lenet_prototype_base.sharpnet",
            Icon = "Network",
            Color = "Cyan",
            TimeAgo = "Вчера, 18:45",
            Status = "Draft",
            StatusColor = "Gray"
        });

        RecentProjects.Add(new RecentProjectViewModel
        {
            Name = "adversarial_testing_01.sharpnet",
            Icon = "ShieldAlert",
            Color = "Rose",
            TimeAgo = "3 дня назад",
            Status = "In Progress",
            StatusColor = "Gold"
        });

        RecentProjects.Add(new RecentProjectViewModel
        {
            Name = "cifar10_baseline.sharpnet",
            Icon = "Layers",
            Color = "Gray",
            TimeAgo = "1 неделю назад",
            Status = "",
            StatusColor = "Gray",
            IsDimmed = true
        });

        RecentProjects.Add(new RecentProjectViewModel
        {
            Name = "transfer_learning_v2.sharpnet",
            Icon = "Cpu",
            Color = "Gray",
            TimeAgo = "2 недели назад",
            Status = "",
            StatusColor = "Gray",
            IsDimmed = true
        });
    }

    private void NewProject(object? parameter) { /* Логика */ }
    private void OpenProject(object? parameter) { /* Логика */ }
    private void OpenExamples(object? parameter) { /* Логика */ }
    private void OpenDatasets(object? parameter) { /* Логика */ }
    private void ShowTutorials(object? parameter) { /* Логика */ }
    private void OpenDocumentation(object? parameter) { /* Логика */ }
    private void OpenSupport(object? parameter) { /* Логика */ }
}
