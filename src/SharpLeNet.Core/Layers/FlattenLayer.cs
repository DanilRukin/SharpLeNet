namespace SharpLeNet.Core.Layers;

/// <summary>
/// Слой Flatten (преобразует многомерный тензор в 2D)
/// </summary>
public class FlattenLayer : Layer
{
    private int _batchSize;
    public override Tensor Forward(Tensor input)
    {
        if (input.Rank < 2)
            throw new ArgumentException("Flatten-слой ожидает как минимум 2D тензор");
        _batchSize = input.Shape[0];
        int features = input.Size / _batchSize;

        return input.Reshape(_batchSize, features);
    }
}
