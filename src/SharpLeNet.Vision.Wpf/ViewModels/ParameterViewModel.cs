using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class ParameterViewModel : BaseViewModel
{
    public string Label { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
}
