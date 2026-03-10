using SharpLeNet.Core;
using SharpLeNet.Vision.Wpf.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLeNet.Vision.Wpf.Services;

public class DataService : IDataService
{
    private readonly ILoggerService _logger;
    private Tensor? _trainImages;
    private Tensor? _trainLabels;
    private Tensor? _testImages;
    private Tensor? _testLabels;

    public bool IsLoaded => _trainImages != null;
    public int TrainingImageCount => _trainImages?.Shape[0] ?? 0;
    public int TestImageCount => _testImages?.Shape[0] ?? 0;

    public DataService(ILoggerService logger)
    {
        _logger = logger;
    }

    public async Task<bool> LoadMNISTAsync(IProgress<double>? progress = null)
    {
        try
        {
            _logger.Info("Loading MNIST dataset...", "DataService");

            // В реальном приложении здесь будет загрузка из файлов
            // Для MVP используем InMemoryMNISTDataset
            await Task.Run(() =>
            {
                // Имитация загрузки с прогрессом
                for (int i = 0; i <= 100; i += 10)
                {
                    Thread.Sleep(100);
                    progress?.Report(i / 100.0);
                }

                // Создаем синтетические данные для демонстрации
                CreateSyntheticData();
            });

            _logger.Info($"MNIST loaded: {TrainingImageCount} training, {TestImageCount} test images", "DataService");
            return true;
        }
        catch (Exception ex)
        {
            _logger.Error($"Failed to load MNIST: {ex.Message}", "DataService");
            return false;
        }
    }

    private void CreateSyntheticData()
    {
        // Для демонстрации создаем синтетические данные
        // В реальном приложении здесь будет загрузка настоящих файлов MNIST

        var random = new Random(42);

        // 60000 тренировочных изображений 28x28
        var trainData = new Tensor([6000, 28, 28]); // Уменьшим для демо
        var trainLabelData = new Tensor([6000, 10]);

        for (int i = 0; i < 6000; i++)
        {
            int label = i % 10;
            trainLabelData[i, label] = 1.0;

            for (int h = 0; h < 28; h++)
                for (int w = 0; w < 28; w++)
                {
                    trainData[i, h, w] = random.NextDouble() * 0.5 + (label * 0.05);
                }
        }

        _trainImages = trainData;
        _trainLabels = trainLabelData;

        // 10000 тестовых изображений
        var testData = new Tensor([1000, 28, 28]);
        var testLabelData = new Tensor([1000, 10]);

        for (int i = 0; i < 1000; i++)
        {
            int label = i % 10;
            testLabelData[i, label] = 1.0;

            for (int h = 0; h < 28; h++)
                for (int w = 0; w < 28; w++)
                {
                    testData[i, h, w] = random.NextDouble() * 0.5 + (label * 0.05);
                }
        }

        _testImages = testData;
        _testLabels = testLabelData;
    }

    public Tensor GetTrainingImages() => _trainImages ?? throw new InvalidOperationException("Data not loaded");
    public Tensor GetTrainingLabels() => _trainLabels ?? throw new InvalidOperationException("Data not loaded");
    public Tensor GetTestImages() => _testImages ?? throw new InvalidOperationException("Data not loaded");
    public Tensor GetTestLabels() => _testLabels ?? throw new InvalidOperationException("Data not loaded");

    public MNISTSample GetSample(int index, bool isTraining = true)
    {
        var images = isTraining ? _trainImages : _testImages;
        var labels = isTraining ? _trainLabels : _testLabels;

        if (images == null || labels == null)
            throw new InvalidOperationException("Data not loaded");

        // Находим метку (one-hot -> число)
        int label = 0;
        for (int i = 0; i < 10; i++)
        {
            if (labels[index, i] > 0.5)
            {
                label = i;
                break;
            }
        }

        return new MNISTSample
        {
            Index = index,
            Label = label,
            Width = 28,
            Height = 28
            // В реальном приложении здесь будет ImageData
        };
    }

    public List<MNISTSample> GetPreviewSamples(int count = 20, bool isTraining = true)
    {
        var samples = new List<MNISTSample>();
        var total = isTraining ? TrainingImageCount : TestImageCount;

        for (int i = 0; i < Math.Min(count, total); i++)
        {
            samples.Add(GetSample(i, isTraining));
        }

        return samples;
    }
}
