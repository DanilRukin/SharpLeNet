namespace SharpLeNet.Core.Optimizers;

/// <summary>
/// Базовый класс оптимизатора
/// </summary>
public abstract class Optimizer
{
    protected readonly List<Tensor> _parameters;
    protected readonly double _learningRate;

    protected Optimizer(List<Tensor> parameters, double learningRate)
    {
        _parameters = parameters;
        _learningRate = learningRate;
    }
    /// <summary>
    /// Шаг оптимизатора
    /// </summary>
    public abstract void Step();

    /// <summary>
    /// Зануляет градиенты
    /// </summary>
    public abstract void ZeroGrad();
}
