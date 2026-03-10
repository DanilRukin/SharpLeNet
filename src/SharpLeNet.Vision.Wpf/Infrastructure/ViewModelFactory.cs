namespace SharpLeNet.Vision.Wpf.Infrastructure;

public class ViewModelFactory : IViewModelFactory
{
    public T Create<T>() where T : BaseViewModel
    {
        return Activator.CreateInstance<T>();
    }
}
