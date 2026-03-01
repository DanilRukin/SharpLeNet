namespace SharpLeNet.Core.Training;

/// <summary>
/// Данные обучения
/// </summary>
public struct TrainingStats
{
    /// <summary>
    /// Потери
    /// </summary>
    public double Loss;

    /// <summary>
    /// Доля правильных ответов
    /// </summary>
    public double Accuracy;
}
