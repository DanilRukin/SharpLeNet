using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class FileMenuViewModel : BaseMenuViewModel
{
    public FileMenuViewModel()
    {
        RecentProjects = new ObservableCollection<RecentFileViewModel>();
        LoadRecentProjects();

        // Команды
        NewProjectCommand = new RelayCommand(NewProject);
        OpenProjectCommand = new RelayCommand(OpenProject);
        SaveProjectCommand = new RelayCommand(SaveProject);
        SaveAsCommand = new RelayCommand(SaveAs);
        ExportModelCommand = new RelayCommand(ExportModel);
        ImportWeightsCommand = new RelayCommand(ImportWeights);
        ProjectSettingsCommand = new RelayCommand(ProjectSettings);
        AppSettingsCommand = new RelayCommand(AppSettings);
        ExitCommand = new RelayCommand(Exit);
    }

    public ObservableCollection<RecentFileViewModel> RecentProjects { get; }

    // Команды
    public ICommand NewProjectCommand { get; }
    public ICommand OpenProjectCommand { get; }
    public ICommand SaveProjectCommand { get; }
    public ICommand SaveAsCommand { get; }
    public ICommand ExportModelCommand { get; }
    public ICommand ImportWeightsCommand { get; }
    public ICommand ProjectSettingsCommand { get; }
    public ICommand AppSettingsCommand { get; }
    public ICommand ExitCommand { get; }

    private void LoadRecentProjects()
    {
        RecentProjects.Add(new RecentFileViewModel
        {
            Name = "mnist_experiment_v3.sharpnet",
            TimeAgo = "2ч назад"
        });

        RecentProjects.Add(new RecentFileViewModel
        {
            Name = "lenet_prototype_base.sharpnet",
            TimeAgo = "Вчера"
        });

        RecentProjects.Add(new RecentFileViewModel
        {
            Name = "adversarial_testing_01.sharpnet",
            TimeAgo = "3 дня назад"
        });
    }

    private void NewProject(object? parameter) { /* Логика */ }
    private void OpenProject(object? parameter) { /* Логика */ }
    private void SaveProject(object? parameter) { /* Логика */ }
    private void SaveAs(object? parameter) { /* Логика */ }
    private void ExportModel(object? parameter) { /* Логика */ }
    private void ImportWeights(object? parameter) { /* Логика */ }
    private void ProjectSettings(object? parameter) { /* Логика */ }
    private void AppSettings(object? parameter) { /* Логика */ }
    private void Exit(object? parameter) { /* Логика */ }
}
