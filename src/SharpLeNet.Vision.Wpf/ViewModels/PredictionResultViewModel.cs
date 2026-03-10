using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class PredictionResultViewModel : BaseViewModel
{
    private string _className = string.Empty;
    private double _probability;
    private bool _isTopPrediction;
    private double _confidence;
    private string? _explanation;
    private bool _showExplanation;

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
            }
        }
    }

    public string ProbabilityDisplay => _probability.ToString("F1") + "%";

    public double ProgressWidth => _probability * 2.8; // Для прогресс-бара

    public bool IsTopPrediction
    {
        get => _isTopPrediction;
        set => SetProperty(ref _isTopPrediction, value);
    }

    public double Confidence
    {
        get => _confidence;
        set => SetProperty(ref _confidence, value);
    }

    public string? Explanation
    {
        get => _explanation;
        set => SetProperty(ref _explanation, value);
    }

    public bool ShowExplanation
    {
        get => _showExplanation;
        set => SetProperty(ref _showExplanation, value);
    }
}
