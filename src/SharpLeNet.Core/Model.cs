using SharpLeNet.Core.Layers;

namespace SharpLeNet.Core;

/// <summary>
/// Класс модели нейростеи
/// </summary>
public class Model
{
    private readonly List<Layer> _layers = new List<Layer>();

    /// <summary>
    /// Параметры модели сети
    /// </summary>
    public List<Tensor> Parameters
    {
        get
        {
            List<Tensor> parameters = new List<Tensor>();
            foreach (var layer in _layers)
            {
                parameters.AddRange(layer.Parameters);
            }
            return parameters;
        }
    }

    /// <summary>
    /// Добавляет слой в модель сети
    /// </summary>
    /// <param name="layer">Слой для добавления</param>
    public Model AddLayer(Layer layer)
    {
        _layers.Add(layer);
        return this;
    }

    /// <summary>
    /// Прямой проход через слои сети
    /// </summary>
    /// <param name="input">Тензор, который необходимо пропустить через модель</param>
    public Tensor Forward(Tensor input)
    {
        Tensor output = input;
        foreach (Layer layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Обнуляет градиенты в слоях сети
    /// </summary>
    public void ZeroGrad()
    {
        foreach (Layer layer in _layers)
        {
            layer.ZeroGrad();
        }
    }
}

