using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class ConfusionMatrixViewModel : BaseViewModel
{
    private int[,] _matrix;
    private int _classCount;
    private ObservableCollection<string> _classLabels;
    private int _selectedClass;

    public ConfusionMatrixViewModel(int classCount)
    {
        _classCount = classCount;
        _matrix = new int[classCount, classCount];
        _classLabels = new ObservableCollection<string>();

        for (int i = 0; i < classCount; i++)
        {
            _classLabels.Add($"Class {i}");
        }

        // Демо-данные
        LoadDemoData();
    }

    public int[,] Matrix
    {
        get => _matrix;
        set => SetProperty(ref _matrix, value);
    }

    public ObservableCollection<string> ClassLabels
    {
        get => _classLabels;
        set => SetProperty(ref _classLabels, value);
    }

    public int SelectedClass
    {
        get => _selectedClass;
        set
        {
            if (SetProperty(ref _selectedClass, value))
            {
                OnPropertyChanged(nameof(SelectedClassData));
            }
        }
    }

    public (int TP, int FP, int FN, int TN) SelectedClassData
    {
        get
        {
            var tp = Matrix[SelectedClass, SelectedClass];
            var fp = 0;
            var fn = 0;
            var tn = 0;

            for (int i = 0; i < _classCount; i++)
            {
                for (int j = 0; j < _classCount; j++)
                {
                    if (i == SelectedClass && j == SelectedClass) continue;
                    if (i == SelectedClass) fn += Matrix[i, j];
                    if (j == SelectedClass) fp += Matrix[i, j];
                    if (i != SelectedClass && j != SelectedClass) tn += Matrix[i, j];
                }
            }

            return (tp, fp, fn, tn);
        }
    }

    public double OverallAccuracy
    {
        get
        {
            int total = 0;
            int correct = 0;
            for (int i = 0; i < _classCount; i++)
            {
                for (int j = 0; j < _classCount; j++)
                {
                    total += Matrix[i, j];
                    if (i == j) correct += Matrix[i, j];
                }
            }
            return total > 0 ? (double)correct / total * 100 : 0;
        }
    }

    private void LoadDemoData()
    {
        var random = new Random(42);
        for (int i = 0; i < _classCount; i++)
        {
            for (int j = 0; j < _classCount; j++)
            {
                if (i == j)
                    _matrix[i, j] = 80 + random.Next(15);
                else
                    _matrix[i, j] = random.Next(5);
            }
        }
    }
}
