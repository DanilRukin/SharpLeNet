using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public class RecentProjectViewModel : BaseViewModel
{
    private string _name = string.Empty;
    private string _icon = string.Empty;
    private string _color = string.Empty;
    private string _timeAgo = string.Empty;
    private string _status = string.Empty;
    private string _statusColor = string.Empty;
    private bool _isDimmed;

    public string Name
    {
        get => _name;
        set => SetProperty(ref _name, value);
    }

    public string Icon
    {
        get => _icon;
        set => SetProperty(ref _icon, value);
    }

    public string Color
    {
        get => _color;
        set => SetProperty(ref _color, value);
    }

    public string TimeAgo
    {
        get => _timeAgo;
        set => SetProperty(ref _timeAgo, value);
    }

    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    public string StatusColor
    {
        get => _statusColor;
        set => SetProperty(ref _statusColor, value);
    }

    public bool IsDimmed
    {
        get => _isDimmed;
        set => SetProperty(ref _isDimmed, value);
    }
}
