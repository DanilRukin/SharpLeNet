using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class RecentFileViewModel : BaseViewModel
{
    private string _name = string.Empty;
    private string _timeAgo = string.Empty;

    public string Name
    {
        get => _name;
        set => SetProperty(ref _name, value);
    }

    public string TimeAgo
    {
        get => _timeAgo;
        set => SetProperty(ref _timeAgo, value);
    }
}
