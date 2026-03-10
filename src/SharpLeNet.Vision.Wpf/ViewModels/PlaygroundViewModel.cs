using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class PlaygroundViewModel : BaseViewModel
{
    private DrawingCanvasViewModel _drawingCanvas;
    private ObservableCollection<PredictionResultViewModel> _predictions;
    private string? _selectedImagePath;
    private bool _isDrawing;
    private bool _showUncertainty = true;
    private int _topKResults = 5;
    private string _selectedModelVersion = "LeNet-5 (trained)";

    public PlaygroundViewModel()
    {
        _drawingCanvas = new DrawingCanvasViewModel(280, 280);
        _predictions = new ObservableCollection<PredictionResultViewModel>();

        AvailableModels = new ObservableCollection<string>
            {
                "LeNet-5 (trained)",
                "LeNet-5 (experimental)",
                "Custom Model v1",
                "Custom Model v2"
            };

        RecentImages = new ObservableCollection<string>
            {
                "demo_0.png",
                "demo_1.png",
                "demo_2.png",
                "demo_3.png",
                "demo_4.png"
            };

        // Команды
        ClearCanvasCommand = new RelayCommand(ClearCanvas);
        PredictCommand = new RelayCommand(Predict, (_) => _drawingCanvas.HasDrawing || _selectedImagePath != null);
        LoadImageCommand = new RelayCommand(LoadImage);
        SaveDrawingCommand = new RelayCommand(SaveDrawing);
        CompareModelsCommand = new RelayCommand(CompareModels);
        ShowExplanationCommand = new RelayCommand<PredictionResultViewModel>(ShowExplanation);

        // Загружаем демо-предсказание
        LoadDemoPredictions();
    }

    public DrawingCanvasViewModel DrawingCanvas
    {
        get => _drawingCanvas;
        set => SetProperty(ref _drawingCanvas, value);
    }

    public ObservableCollection<PredictionResultViewModel> Predictions
    {
        get => _predictions;
        set => SetProperty(ref _predictions, value);
    }

    public ObservableCollection<string> AvailableModels { get; }
    public ObservableCollection<string> RecentImages { get; }

    public string? SelectedImagePath
    {
        get => _selectedImagePath;
        set
        {
            if (SetProperty(ref _selectedImagePath, value))
            {
                OnPropertyChanged(nameof(CanPredict));
                LoadSelectedImage();
            }
        }
    }

    public string SelectedModelVersion
    {
        get => _selectedModelVersion;
        set => SetProperty(ref _selectedModelVersion, value);
    }

    public bool IsDrawing
    {
        get => _isDrawing;
        set => SetProperty(ref _isDrawing, value);
    }

    public bool ShowUncertainty
    {
        get => _showUncertainty;
        set => SetProperty(ref _showUncertainty, value);
    }

    public int TopKResults
    {
        get => _topKResults;
        set
        {
            if (SetProperty(ref _topKResults, value))
            {
                UpdatePredictionsDisplay();
            }
        }
    }

    public bool CanPredict => _drawingCanvas.HasDrawing || _selectedImagePath != null;

    public PredictionResultViewModel? TopPrediction =>
        Predictions.Count > 0 ? Predictions[0] : null;

    // Команды
    public ICommand ClearCanvasCommand { get; }
    public ICommand PredictCommand { get; }
    public ICommand LoadImageCommand { get; }
    public ICommand SaveDrawingCommand { get; }
    public ICommand CompareModelsCommand { get; }
    public ICommand ShowExplanationCommand { get; }

    private void ClearCanvas(object? parameter)
    {
        _drawingCanvas.Clear();
        Predictions.Clear();
    }

    private void Predict(object? parameter)
    {
        // В реальном приложении - запуск модели на изображении
        // Сейчас просто генерируем демо-данные
        GenerateRandomPredictions();
    }

    private void LoadImage(object? parameter)
    {
        // В реальном приложении - диалог выбора файла
        SelectedImagePath = "selected_image.png";
        IsDrawing = false;
    }

    private void SaveDrawing(object? parameter)
    {
        // Сохранение рисунка
        _drawingCanvas.Save("drawing.png");
    }

    private void CompareModels(object? parameter)
    {
        // Сравнение нескольких моделей
    }

    private void ShowExplanation(PredictionResultViewModel? prediction)
    {
        if (prediction != null)
        {
            // Показать Grad-CAM или другое объяснение
            prediction.ShowExplanation = !prediction.ShowExplanation;
        }
    }

    private void LoadSelectedImage()
    {
        if (_selectedImagePath != null)
        {
            IsDrawing = false;
            // Загрузить изображение в канвас
        }
    }

    private void GenerateRandomPredictions()
    {
        Predictions.Clear();
        var random = new Random(42);
        var classes = new[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

        // Генерируем случайные вероятности
        var probs = new double[10];
        double sum = 0;
        for (int i = 0; i < 10; i++)
        {
            probs[i] = random.NextDouble();
            sum += probs[i];
        }

        // Нормализуем
        for (int i = 0; i < 10; i++)
        {
            probs[i] = probs[i] / sum * 100;
        }

        // Сортируем и берем Top-K
        var indices = new int[10];
        for (int i = 0; i < 10; i++) indices[i] = i;
        Array.Sort(probs, indices);
        Array.Reverse(probs);
        Array.Reverse(indices);

        for (int i = 0; i < Math.Min(TopKResults, 10); i++)
        {
            Predictions.Add(new PredictionResultViewModel
            {
                ClassName = $"Digit {classes[indices[i]]}",
                Probability = probs[i],
                IsTopPrediction = i == 0,
                Confidence = probs[i] / 100,
                Explanation = GenerateExplanation(indices[i])
            });
        }
    }

    private string GenerateExplanation(int classIndex)
    {
        return classIndex switch
        {
            0 => "Сеть видит замкнутый контур в верхней части",
            1 => "Вертикальная линия справа является ключевым признаком",
            2 => "Сеть обращает внимание на петлю в верхней левой части",
            3 => "Две петли и соединение между ними",
            4 => "Открытая петля с вертикальной линией",
            5 => "Сеть анализирует нижнюю петлю и верхнюю дугу",
            6 => "Петля в верхней части с наклоном",
            7 => "Горизонтальная линия сверху - основной признак",
            8 => "Две петли, одна над другой",
            9 => "Петля в верхней части и вертикальная линия",
            _ => "Анализ активаций показывает характерные паттерны"
        };
    }

    private void LoadDemoPredictions()
    {
        GenerateRandomPredictions();
    }

    private void UpdatePredictionsDisplay()
    {
        if (Predictions.Count > 0)
        {
            GenerateRandomPredictions();
        }
    }
}

