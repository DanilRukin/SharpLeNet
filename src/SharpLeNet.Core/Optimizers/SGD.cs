namespace SharpLeNet.Core.Optimizers;

/// <summary>
/// Стохастический градиентный спуск
/// </summary>
public class SGD : Optimizer
{
    private readonly double _momentum;
    private readonly List<Tensor> _velocities;

    public SGD(List<Tensor> parameters, double learningRate = 0.01, double momentum = 0.0) :
        base(parameters, learningRate)
    {
        _momentum = momentum;
        _velocities = new List<Tensor>();

        foreach (var param in parameters)
        {
            _velocities.Add(new Tensor(param.Shape));
        }
    }

    public override void Step()
    {
        for (int i = 0; i < _parameters.Count; i++)
        {
            Tensor param = _parameters[i];
            Tensor? grad = param.Grad;

            if (grad == null) continue;

            if (_momentum > 0)
            {
                // v = momentum * v - lr * grad
                // param += v
                Tensor velocity = _velocities[i];
                
                for (int j = 0; j < param.Size; j++)
                {
                    velocity.Data[j] = _momentum * velocity.Data[j] - _learningRate * grad.Data[j];
                    param.Data[j] += velocity.Data[j];
                }
            }
            else
            {
                // param -= lr * grad
                for (int j = 0; j < param.Size; j++)
                {
                    param.Data[j] -= _learningRate * grad.Data[j];
                }
            }
        }
    }

    public override void ZeroGrad()
    {
        foreach (var param in _parameters)
        {
            param.ZeroGrad();
        }
    }
}
