namespace SharpLeNet.Core.Losses;

/// <summary>
/// Базовый класс функции потерь
/// </summary>
public abstract class Loss
{
    /// <summary>
    /// Вычисляет значение функции
    /// </summary>
    /// <param name="predictions">Предсказанные значения</param>
    /// <param name="targets">Целевые значения</param>
    public abstract Tensor Compute(Tensor predictions, Tensor targets);

    /// <summary>
    /// Вычисляет значение функции и ее производной
    /// </summary>
    /// <param name="predictions">Предсказанные значения</param>
    /// <param name="targets">Целевые значения</param>
    public abstract (Tensor loss, Tensor grad) ComputeWithGrad(Tensor predictions, Tensor targets);
}
