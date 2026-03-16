using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.ViewModels.Playground;

public class ClassProbabilityViewModel : BaseViewModel
{
    private string _className = string.Empty;
    private double _probability;
    private bool _isWinner;

    public string ClassName
    {
        get => _className;
        set => SetProperty(ref _className, value);
    }

    public double Probability
    {
        get => _probability;
        set
        {
            if (SetProperty(ref _probability, value))
            {
                OnPropertyChanged(nameof(ProbabilityDisplay));
                OnPropertyChanged(nameof(ProgressWidth));
                OnPropertyChanged(nameof(BarColor));
                OnPropertyChanged(nameof(BarStyle));
                OnPropertyChanged(nameof(TextColor));
                OnPropertyChanged(nameof(ValueColor));
                OnPropertyChanged(nameof(FontWeight));
            }
        }
    }

    public bool IsWinner
    {
        get => _isWinner;
        set
        {
            if (SetProperty(ref _isWinner, value))
            {
                OnPropertyChanged(nameof(BarColor));
                OnPropertyChanged(nameof(BarStyle));
                OnPropertyChanged(nameof(TextColor));
                OnPropertyChanged(nameof(ValueColor));
                OnPropertyChanged(nameof(FontWeight));
            }
        }
    }

    public string ProbabilityDisplay => Probability.ToString("F3");

    public double ProgressWidth => Probability * 100;

    public Brush BarColor
    {
        get
        {
            if (IsWinner)
                return (Brush)Application.Current.FindResource("Emerald400Brush");
            return (Brush)Application.Current.FindResource("White20Brush");
        }
    }

    public object BarStyle
    {
        get
        {
            if (IsWinner)
                return Application.Current.FindResource("WinnerBarStyle");
            return Application.Current.FindResource("ProbabilityBarStyle");
        }
    }

    public Brush TextColor
    {
        get
        {
            if (IsWinner)
                return (Brush)Application.Current.FindResource("Emerald400Brush");
            return (Brush)Application.Current.FindResource("White70Brush");
        }
    }

    public Brush ValueColor
    {
        get
        {
            if (IsWinner)
                return (Brush)Application.Current.FindResource("WhiteBrush");
            return (Brush)Application.Current.FindResource("White40Brush");
        }
    }

    public FontWeight FontWeight
    {
        get
        {
            if (IsWinner)
                return FontWeights.Bold;
            return FontWeights.Normal;
        }
    }
}
