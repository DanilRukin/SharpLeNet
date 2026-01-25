namespace SharpLeNet.Core;

/// <summary>
/// Операция над тензором
/// </summary>
public enum TensorOperation
{
    /// <summary>
    /// Дефолтное значение
    /// </summary>
    None,
    /// <summary>
    /// Добавление
    /// </summary>
    Add,

    /// <summary>
    /// Умножение (поэлементное)
    /// </summary>
    Mul,

    /// <summary>
    /// Матричное умножение
    /// </summary>
    MatMul,

    /// <summary>
    /// Сигомоида
    /// </summary>
    Sigmoid,

    /// <summary>
    /// Операция отрицания
    /// </summary>
    Neg,

    /// <summary>
    /// Операция суммирования всех элементов тензора (в результате - скаляр)
    /// </summary>
    Sum
}
