using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.ViewModels.Playground;

public class PlaygroundViewModel : BaseViewModel
{
    private string _selectedInputMethod = "drawing";
    private double _confidenceThreshold = 0.5;
    private bool _normalizationEnabled = true;
    private bool _adversarialMode = false;
    private string _predictedClass = "8";
    private double _confidence = 99.42;
    private double _inferenceTime = 4.2;
    private string _device = "GPU 0";
    private double _entropy = 0.012;
    private string _modelInsight = "Модель видит замкнутые контуры в верхней и нижней части изображения, что подтверждает структуру цифры «8».";
    private string _predictedClassName = "Цифра 8";
    private string _activeLayerName = "Conv2D_Layer_1";


    public PlaygroundViewModel()
    {
        // Инициализация коллекций
        _classProbabilities = new ObservableCollection<ClassProbabilityViewModel>();

        // Команды
        RunModelCommand = new RelayCommand(RunModel, (_) => true);
        ClearCanvasCommand = new RelayCommand(ClearCanvas);
        ExportJsonCommand = new RelayCommand(ExportJson);
        AddToTrainingLogCommand = new RelayCommand(AddToTrainingLog);
        SelectDrawingModeCommand = new RelayCommand(() => SelectInputMethod("drawing"));
        SelectFileUploadCommand = new RelayCommand(() => SelectInputMethod("file"));
        SelectDatasetModeCommand = new RelayCommand(() => SelectInputMethod("dataset"));
        // Загружаем демо-данные
        LoadDemoProbabilities();
    }

    // Свойства
    public string PredictedClass
    {
        get => _predictedClass;
        set => SetProperty(ref _predictedClass, value);
    }

    public string PredictedClassName
    {
        get => _predictedClassName;
        set => SetProperty(ref _predictedClassName, value);
    }

    public double Confidence
    {
        get => _confidence;
        set
        {
            if (SetProperty(ref _confidence, value))
                OnPropertyChanged(nameof(ConfidenceDisplay));
        }
    }

    public string ConfidenceDisplay => $"{Confidence:F2}%";

    public double InferenceTime
    {
        get => _inferenceTime;
        set => SetProperty(ref _inferenceTime, value);
    }

    public string Device
    {
        get => _device;
        set => SetProperty(ref _device, value);
    }

    public double Entropy
    {
        get => _entropy;
        set => SetProperty(ref _entropy, value);
    }

    public string ActiveLayerName
    {
        get => _activeLayerName;
        set => SetProperty(ref _activeLayerName, value);
    }

    // Обновим команду ExportJson
    private void ExportJson(object? parameter)
    {
        // В реальном приложении здесь будет экспорт результатов
        System.Diagnostics.Debug.WriteLine("Exporting results to JSON...");
    }


    public string SelectedInputMethod
    {
        get => _selectedInputMethod;
        set => SetProperty(ref _selectedInputMethod, value);
    }

    public double ConfidenceThreshold
    {
        get => _confidenceThreshold;
        set => SetProperty(ref _confidenceThreshold, value);
    }

    public string ModelInsight
    {
        get => _modelInsight;
        set => SetProperty(ref _modelInsight, value);
    }

    private ObservableCollection<ClassProbabilityViewModel> _classProbabilities;

    public ObservableCollection<ClassProbabilityViewModel> ClassProbabilities
    {
        get => _classProbabilities;
        set => SetProperty(ref _classProbabilities, value);
    }
    // Команды
    public ICommand RunModelCommand { get; }
    public ICommand ClearCanvasCommand { get; }
    public ICommand ExportJsonCommand { get; }
    public ICommand AddToTrainingLogCommand { get; }
    public ICommand SelectDrawingModeCommand { get; }
    public ICommand SelectFileUploadCommand { get; }
    public ICommand SelectDatasetModeCommand { get; }

    private void RunModel(object? parameter)
    {
        // Симуляция запуска модели
        // В реальном приложении здесь будет вызов бэкенда
    }

    private void ClearCanvas(object? parameter)
    {
        // Очистка холста
    }

    private void AddToTrainingLog(object? parameter)
    {
        // В реальном приложении здесь будет добавление в лог
        System.Diagnostics.Debug.WriteLine("Adding to training log...");
    }

    private void LoadDemoProbabilities()
    {
        ClassProbabilities.Clear();

        // Класс 8 (победитель)
        ClassProbabilities.Add(new ClassProbabilityViewModel
        {
            ClassName = "Digit 8",
            Probability = 0.994,
            IsWinner = true
        });

        // Класс 3 (второе место)
        ClassProbabilities.Add(new ClassProbabilityViewModel
        {
            ClassName = "Digit 3",
            Probability = 0.003,
            IsWinner = false
        });

        // Остальные классы
        int[] digits = { 0, 1, 2, 4, 5, 6, 7, 9 };
        foreach (var digit in digits)
        {
            ClassProbabilities.Add(new ClassProbabilityViewModel
            {
                ClassName = $"Digit {digit}",
                Probability = 0.0001,
                IsWinner = false
            });
        }
    }

    private void SelectInputMethod(string method)
    {
        SelectedInputMethod = method;
        // Обновление UI для активной кнопки
        OnPropertyChanged(nameof(IsDrawingMode));
        OnPropertyChanged(nameof(IsFileUploadMode));
        OnPropertyChanged(nameof(IsDatasetMode));
    }

    // Properties для стилей кнопок
    public bool IsDrawingMode => SelectedInputMethod == "drawing";
    public bool IsFileUploadMode => SelectedInputMethod == "file";
    public bool IsDatasetMode => SelectedInputMethod == "dataset";


    public bool NormalizationEnabled
    {
        get => _normalizationEnabled;
        set
        {
            if (SetProperty(ref _normalizationEnabled, value))
            {
                Debug.WriteLine($"Normalization toggled to: {value}");
            }
        }
    }

    public bool AdversarialMode
    {
        get => _adversarialMode;
        set
        {
            if (SetProperty(ref _adversarialMode, value))
            {
                Debug.WriteLine($"Adversarial toggled to: {value}");
            }
        }
    }

}

