namespace SharpLeNet.Core.Layers;

/// <summary>
/// Слой Softmax
/// </summary>
public class SoftmaxLayer : Layer
{
    public override Tensor Forward(Tensor input)
    {
        return input.Softmax();
    }
}
