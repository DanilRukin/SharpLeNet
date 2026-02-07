namespace SharpLeNet.Core.Layers;

/// <summary>
/// Слой активации ReLU
/// </summary>
public class ReLULayer : Layer
{
    public override Tensor Forward(Tensor input)
    {
        return input.ReLU();
    }
}
