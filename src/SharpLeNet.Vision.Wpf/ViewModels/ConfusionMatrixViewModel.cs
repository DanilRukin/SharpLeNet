using SharpLeNet.Vision.Wpf.Infrastructure;
using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class ConfusionMatrixViewModel : BaseViewModel
{
    private int _selectedClassIndex = 0;
    private ObservableCollection<MatrixCell> _matrixCells;
    private ObservableCollection<string> _classLabels;

    public ConfusionMatrixViewModel(int classCount)
    {
        ClassLabels = new ObservableCollection<string>();
        for (int i = 0; i < classCount; i++)
        {
            ClassLabels.Add(i.ToString());
        }

        _matrixCells = new ObservableCollection<MatrixCell>();
        InitializeMatrix();
    }

    public ObservableCollection<string> ClassLabels
    {
        get => _classLabels;
        set => SetProperty(ref _classLabels, value);
    }

    public ObservableCollection<MatrixCell> MatrixCells
    {
        get => _matrixCells;
        set => SetProperty(ref _matrixCells, value);
    }

    public int SelectedClassIndex
    {
        get => _selectedClassIndex;
        set
        {
            if (SetProperty(ref _selectedClassIndex, value))
            {
                OnPropertyChanged(nameof(SelectedClassData));
            }
        }
    }

    public string SelectedClass
    {
        get => ClassLabels[SelectedClassIndex];
        set
        {
            int index = ClassLabels.IndexOf(value);
            if (index >= 0)
            {
                SelectedClassIndex = index;
            }
        }
    }

    public ClassMetrics SelectedClassData
    {
        get
        {
            // Здесь будет реальный расчет на основе матрицы
            return new ClassMetrics
            {
                TP = 82,
                FP = 3,
                FN = 5,
                TN = 890
            };
        }
    }

    public double OverallAccuracy
    {
        get
        {
            // Здесь будет реальный расчет
            return 98.1;
        }
    }

    public string OverallAccuracyDisplay => $"{OverallAccuracy:F1}%";

    private void InitializeMatrix()
    {
        var random = new Random(42);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                int value;
                if (i == j)
                {
                    value = 80 + random.Next(15);
                }
                else
                {
                    value = random.Next(8);
                }

                var color = i == j
                    ? new SolidColorBrush(Color.FromRgb(212, 175, 55)) // Gold
                    : new SolidColorBrush(Color.FromRgb(50, 50, 55)); // Dark gray

                _matrixCells.Add(new MatrixCell
                {
                    Row = i,
                    Column = j,
                    Value = value,
                    Color = color,
                    TextColor = i == j ? Brushes.Black : Brushes.White,
                    ToolTip = $"Actual: {i}, Predicted: {j}\nCount: {value}"
                });
            }
        }
    }
}

