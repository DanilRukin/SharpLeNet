namespace SharpLeNet.Vision.Wpf.Infrastructure;

public interface IViewModelFactory
{
    T Create<T>() where T : BaseViewModel;
}
