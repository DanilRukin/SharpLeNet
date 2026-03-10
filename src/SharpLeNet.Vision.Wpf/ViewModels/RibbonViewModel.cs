using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class RibbonViewModel : BaseViewModel
{
    private readonly MainViewModel _mainViewModel;
    private string _searchText = string.Empty;
    private bool _isTraining;

    public RibbonViewModel(MainViewModel mainViewModel)
    {
        _mainViewModel = mainViewModel ?? throw new ArgumentNullException(nameof(mainViewModel));

        // Команды меню
        NavigateHomeCommand = new RelayCommand(NavigateHome);
        ShowFileMenuCommand = new RelayCommand(ShowFileMenu);
        ShowViewMenuCommand = new RelayCommand(ShowViewMenu);
        ShowHelpCommand = new RelayCommand(ShowHelp);

        // Команды действий
        QuickStartCommand = new RelayCommand(QuickStart);
        OpenSettingsCommand = new RelayCommand(OpenSettings);

        // Команды управления обучением (пробрасываем в MainViewModel)
        StartTrainingCommand = new RelayCommand(_ => _mainViewModel.StartTrainingCommand.Execute(null),
                                               _ => _mainViewModel.StartTrainingCommand.CanExecute(null));
        PauseTrainingCommand = new RelayCommand(_ => _mainViewModel.PauseTrainingCommand.Execute(null),
                                               _ => _mainViewModel.PauseTrainingCommand.CanExecute(null));
        StopTrainingCommand = new RelayCommand(_ => _mainViewModel.StopTrainingCommand.Execute(null),
                                              _ => _mainViewModel.StopTrainingCommand.CanExecute(null));
    }

    // Свойства
    public string SearchText
    {
        get => _searchText;
        set
        {
            if (SetProperty(ref _searchText, value))
            {
                // Здесь можно добавить логику поиска
                OnSearchTextChanged(value);
            }
        }
    }

    public string ProjectName => _mainViewModel.ProjectName;
    public string CurrentMode => _mainViewModel.CurrentMode;
    public bool IsTraining
    {
        get => _isTraining;
        set => SetProperty(ref _isTraining, value);
    }

    // Команды
    public ICommand NavigateHomeCommand { get; }
    public ICommand ShowFileMenuCommand { get; }
    public ICommand ShowViewMenuCommand { get; }
    public ICommand ShowHelpCommand { get; }
    public ICommand QuickStartCommand { get; }
    public ICommand OpenSettingsCommand { get; }
    public ICommand StartTrainingCommand { get; }
    public ICommand PauseTrainingCommand { get; }
    public ICommand StopTrainingCommand { get; }

    // Методы команд
    private void NavigateHome(object? parameter)
    {
        // Навигация на главный экран
        _mainViewModel.SelectedTab = _mainViewModel.Architecture;
    }

    private void ShowFileMenu(object? parameter)
    {
        // Показать меню File
        var dialog = new System.Windows.Controls.ContextMenu();
        // Здесь можно создать меню программно или через XAML
    }

    private void ShowViewMenu(object? parameter)
    {
        // Показать меню View
    }

    private void ShowHelp(object? parameter)
    {
        // Показать справку
        System.Windows.MessageBox.Show("SharpLeNet Vision - Визуальная лаборатория компьютерного зрения\n\n" +
                                      "Версия 1.0.0\n\n" +
                                      "Документация: https://github.com/sharplenet/vision",
                                      "О программе",
                                      System.Windows.MessageBoxButton.OK,
                                      System.Windows.MessageBoxImage.Information);
    }

    private void QuickStart(object? parameter)
    {
        _mainViewModel.QuickStartCommand.Execute(parameter);
    }

    private void OpenSettings(object? parameter)
    {
        // Открыть окно настроек
        // В реальном приложении здесь будет вызов диалога настроек
        System.Windows.MessageBox.Show("Настройки приложения", "Settings",
                                      System.Windows.MessageBoxButton.OK,
                                      System.Windows.MessageBoxImage.Information);
    }

    private void OnSearchTextChanged(string text)
    {
        // Логика поиска
        if (!string.IsNullOrWhiteSpace(text))
        {
            // Поиск по слоям, параметрам и т.д.
        }
    }

    // Обновление состояния
    public void UpdateTrainingState(bool isTraining)
    {
        IsTraining = isTraining;
    }
}
