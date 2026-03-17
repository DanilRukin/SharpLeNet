using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class HelpMenuViewModel : BaseMenuViewModel
{
    private string _appVersion = "v1.4.2";

    public HelpMenuViewModel()
    {
        // Команды
        QuickStartCommand = new RelayCommand(QuickStart);
        UserGuideCommand = new RelayCommand(UserGuide);
        ArchitectureReferenceCommand = new RelayCommand(ArchitectureReference);

        VideoTutorialsCommand = new RelayCommand(VideoTutorials);
        KeyboardShortcutsCommand = new RelayCommand(KeyboardShortcuts);
        ProTipsCommand = new RelayCommand(ProTips);

        ReportBugCommand = new RelayCommand(ReportBug);
        OpenGitHubCommand = new RelayCommand(OpenGitHub);
        JoinDiscordCommand = new RelayCommand(JoinDiscord);
        SuggestFeatureCommand = new RelayCommand(SuggestFeature);

        AboutCommand = new RelayCommand(About);
        CheckUpdatesCommand = new RelayCommand(CheckUpdates);
    }

    public string AppVersion
    {
        get => _appVersion;
        set => SetProperty(ref _appVersion, value);
    }

    // Documentation & Learning
    public ICommand QuickStartCommand { get; }
    public ICommand UserGuideCommand { get; }
    public ICommand ArchitectureReferenceCommand { get; }

    // Tutorials & Resources
    public ICommand VideoTutorialsCommand { get; }
    public ICommand KeyboardShortcutsCommand { get; }
    public ICommand ProTipsCommand { get; }

    // Community & Support
    public ICommand ReportBugCommand { get; }
    public ICommand OpenGitHubCommand { get; }
    public ICommand JoinDiscordCommand { get; }
    public ICommand SuggestFeatureCommand { get; }

    // About
    public ICommand AboutCommand { get; }
    public ICommand CheckUpdatesCommand { get; }

    private void QuickStart(object? parameter) { /* Логика */ }
    private void UserGuide(object? parameter) { /* Логика */ }
    private void ArchitectureReference(object? parameter) { /* Логика */ }
    private void VideoTutorials(object? parameter) { /* Логика */ }
    private void KeyboardShortcuts(object? parameter) { /* Логика */ }
    private void ProTips(object? parameter) { /* Логика */ }
    private void ReportBug(object? parameter) { /* Логика */ }
    private void OpenGitHub(object? parameter) { /* Логика */ }
    private void JoinDiscord(object? parameter) { /* Логика */ }
    private void SuggestFeature(object? parameter) { /* Логика */ }
    private void About(object? parameter) { /* Логика */ }
    private void CheckUpdates(object? parameter) { /* Логика */ }
}
