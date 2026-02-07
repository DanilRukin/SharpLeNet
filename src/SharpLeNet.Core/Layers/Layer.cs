namespace SharpLeNet.Core.Layers;

/// <summary>
/// Базовый класс слоя
/// </summary>
public abstract class Layer
{
    /// <summary>
    /// Параметры (веса, смещения)
    /// </summary>
    public List<Tensor> Parameters { get; protected set; } = new List<Tensor>();

    /// <summary>
    /// Прямой проход
    /// </summary>
    /// <param name="input">Входной тензор</param>
    public abstract Tensor Forward(Tensor input);

    /// <summary>
    /// Обнуляет градиенты. Обратный проход
    /// </summary>
    public virtual void ZeroGrad()
    {
        foreach (Tensor param in Parameters)
        {
            param.ZeroGrad();
        }
    }
}
