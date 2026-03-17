using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Menu;

public abstract class BaseMenuViewModel : BaseViewModel
{
    private bool _isOpen;

    public bool IsOpen
    {
        get => _isOpen;
        set => SetProperty(ref _isOpen, value);
    }

    public ICommand CloseCommand { get; }

    protected BaseMenuViewModel()
    {
        CloseCommand = new RelayCommand(Close);
    }

    public virtual void Open()
    {
        IsOpen = true;
    }

    private void Close(object? parameter)
    {
        IsOpen = false;
    }
}