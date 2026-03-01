using SharpLeNet.Core.Data;
using SharpLeNet.Core.Losses;
using SharpLeNet.Core.Optimizers;

namespace SharpLeNet.Core.Training;

/// <summary>
/// Главный класс для обучения моделей
/// </summary>
public class Trainer
{
    private readonly Model _model;
    private readonly Optimizer _optimizer;
    private readonly Loss _loss;
    private readonly DataLoader _trainLoader;
    private readonly DataLoader? _valLoader;
    private readonly List<Callback> _callbacks;

    private List<double> _trainLosses;
    private List<double> _valLosses;
    private List<double> _trainAccuracies;
    private List<double> _valAccuracies;

    public IReadOnlyList<double> TrainLosses => _trainLosses.AsReadOnly();
    public IReadOnlyList<double> ValLosses => _valLosses.AsReadOnly();
    public IReadOnlyList<double> TrainAccuracies => _trainAccuracies.AsReadOnly();
    public IReadOnlyList<double> ValAccuracies => _valAccuracies.AsReadOnly();

    /// <summary>
    /// Конструктор тренера
    /// </summary>
    /// <param name="model">Модель для обучения</param>
    /// <param name="optimizer">Оптимизатор</param>
    /// <param name="loss">Функция потерь</param>
    /// <param name="trainLoader">DataLoader для тренировочных данных</param>
    /// <param name="valLoader">DataLoader для валидационных данных (опционально)</param>
    public Trainer(Model model, Optimizer optimizer, Loss loss,
                   DataLoader trainLoader, DataLoader? valLoader = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
        _trainLoader = trainLoader ?? throw new ArgumentNullException(nameof(trainLoader));
        _valLoader = valLoader;

        _callbacks = new List<Callback>();
        _trainLosses = new List<double>();
        _valLosses = new List<double>();
        _trainAccuracies = new List<double>();
        _valAccuracies = new List<double>();
    }

    public void AddCallback(Callback callback)
    {
        _callbacks.Add(callback);
    }

    /// <summary>
    /// Одна эпоха обучения
    /// </summary>
    private TrainingStats TrainEpoch()
    {
        double totalLoss = 0;
        double totalAcc = 0;
        int batches = 0;

        foreach (var (batchFeatures, batchLabels) in _trainLoader)
        {
            // Прямой проход
            Tensor predictions = _model.Forward(batchFeatures);
            Tensor loss = _loss.Compute(predictions, batchLabels);

            // Считаем метрики
            double lossValue = loss.Data[0];
            double accValue = Metrics.Metrics.Accuracy(predictions, batchLabels);

            totalLoss += lossValue;
            totalAcc += accValue;
            batches++;

            // Обратный проход
            _optimizer.ZeroGrad();
            loss.Backward();
            _optimizer.Step();

            // Вызываем callback для батча
            foreach (Callback callback in _callbacks)
            {
                callback.OnBatchEnd(batches, lossValue);
            }
        }

        return new TrainingStats
        {
            Loss = totalLoss / batches,
            Accuracy = totalAcc / batches,
        };
    }

    /// <summary>
    /// Валидация на эпохе
    /// </summary>
    private TrainingStats ValidateEpoch()
    {
        if (_valLoader == null)
            return new TrainingStats { Loss = 0, Accuracy = 0 };

        double totalLoss = 0;
        double totalAcc = 0;
        int batches = 0;

        foreach (var (batchFeatures, batchLabels) in _valLoader)
        {
            Tensor predictions = _model.Forward(batchFeatures);
            Tensor loss = _loss.Compute(predictions, batchLabels);

            totalLoss += loss.Data[0];
            totalAcc += Metrics.Metrics.Accuracy(predictions, batchLabels);
            batches++;
        }

        return new TrainingStats
        {
            Loss = totalLoss / batches,
            Accuracy = totalAcc / batches
        };
    }

    /// <summary>
    /// Запуск обучения
    /// </summary>
    /// <param name="epochs">Кол-во эпох</param>
    public void Fit(int epochs)
    {
        Console.WriteLine("=== Начало обучения ===");
        Console.WriteLine($"Модель: {_model.GetType().Name}");
        Console.WriteLine($"Оптимизатор: {_optimizer.GetType().Name}");
        Console.WriteLine($"Функция потерь: {_loss.GetType().Name}");
        Console.WriteLine($"Эпох: {epochs}");
        Console.WriteLine($"Train samples: {_trainLoader.Count() * _trainLoader.First().features.Shape[0]}");
        Console.WriteLine();

        // Начало обучения
        foreach (Callback callback in _callbacks)
            callback.OnTrainBegin();

        bool isEarlyStopping = false;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Начало эпохи
            foreach (Callback callback in _callbacks)
                callback.OnEpochBegin(epoch);

            // Сбрасываем состояние DataLoader'ов для новой эпохи
            // (в нашей реализации сброс происходит автоматически при каждой итерации)

            // Обучение
            TrainingStats trainingStats = TrainEpoch();
            _trainLosses.Add(trainingStats.Loss);
            _trainAccuracies.Add(trainingStats.Accuracy);

            // Валидация
            TrainingStats valStats = ValidateEpoch();
            _valLosses.Add(valStats.Loss);
            _valAccuracies.Add(valStats.Accuracy);

            // Вывод прогресса
            Metrics.Metrics.PrintProgress(epoch, epochs,
                trainingStats.Loss, trainingStats.Accuracy,
                valStats.Loss, valStats.Accuracy);

            foreach (Callback callback in _callbacks)
            {
                callback.OnEpochEnd(epoch, trainingStats.Loss, valStats.Loss,
                    trainingStats.Accuracy, valStats.Accuracy);

                // Проверка на раннюю остановку
                if (callback is EarlyStopping es && es.ShouldStop)
                {
                    Console.WriteLine("Обучение остановлено досрочно");
                    isEarlyStopping = true;
                    break;
                }
            }
            if (isEarlyStopping) break;
        }

        // Конец обучения
        foreach (var callback in _callbacks)
            callback.OnTrainEnd();

        Console.WriteLine("\n=== Обучение завершено ===");
    }
}