namespace SharpLeNet.Core.Metrics;

/// <summary>
/// Метрики для оценки качества модели
/// </summary>
public class Metrics
{
    /// <summary>
    /// Вычисляет accuracy (долю правильных ответов)
    /// </summary>
    public static double Accuracy(Tensor predictions, Tensor targets)
    {
        if (predictions.Rank != 2 || targets.Rank != 2)
            throw new ArgumentException("Ожидаются двумерные тензоры [batch, classes]");

        int batchSize = predictions.Shape[0];
        int numClasses = predictions.Shape[1];

        int correct = 0;
        for (int i = 0; i < batchSize; i++)
        {
            // Находим предсказанный класс (максимальная вероятность)
            int predClass = 0;
            double maxProb = double.MinValue;
            for (int j = 0; j < numClasses; j++)
            {
                if (predictions[i, j] > maxProb)
                {
                    maxProb = predictions[i, j];
                    predClass = j;
                }
            }

            // Находим истинный класс (где 1.0 в one-hot)
            int trueClass = 0;
            for (int j = 0; j < numClasses; j++)
            {
                if (Math.Abs(targets[i, j] - 1.0) < 1e-6)
                {
                    trueClass = j;
                    break;
                }
            }

            if (predClass == trueClass)
                correct++;
        }

        return (double)correct / batchSize;
    }

    /// <summary>
    /// Среднее значение потерь
    /// </summary>
    public static double MeanLoss(List<double> losses)
    {
        return losses.Count > 0 ? losses.Average() : 0.0;
    }

    /// <summary>
    /// Выводит прогресс-бар обучения
    /// </summary>
    public static void PrintProgress(int epoch, int totalEpochs,
        double trainLoss, double trainAcc, double valLoss, double valAcc)
    {
        Console.WriteLine(
            $"Epoch [{epoch + 1}/{totalEpochs}] | " +
            $"Train Loss: {trainLoss:F6} | Train Acc: {trainAcc:F4} | " +
            $"Val Loss: {valLoss:F6} | Val Acc: {valAcc:F4}");
    }
}

