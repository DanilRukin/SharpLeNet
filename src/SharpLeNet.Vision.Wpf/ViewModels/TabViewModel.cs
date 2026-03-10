using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class TabViewModel : BaseViewModel
{
    private bool _isActive;

    public TabViewModel(string name, Type viewModelType, bool isActive = false)
    {
        Name = name;
        ViewModelType = viewModelType;
        IsActive = isActive;
    }

    public string Name { get; }
    public Type ViewModelType { get; }

    public bool IsActive
    {
        get => _isActive;
        set => SetProperty(ref _isActive, value);
    }
}
