namespace SharpLeNet.Core.Losses;

/// <summary>
/// Mean Squared Error Loss
/// </summary>
public class MSELoss : Loss
{
    private readonly bool _reductionMean;

    public MSELoss(bool reductionMean = true)
    {
        _reductionMean = reductionMean;
    }

    public override Tensor Compute(Tensor predictions, Tensor targets)
    {
        if (!predictions.Shape.SequenceEqual(targets.Shape))
            throw new ArgumentException("Предсказания и цели должны иметь одинаковые размерности");
        Tensor diff = predictions - targets;
        Tensor square = diff * diff;
        if (_reductionMean)
        {
            return square.Sum() / predictions.Size;
        }
        else
        {
            return square.Sum();
        }
    }

    public override (Tensor loss, Tensor grad) ComputeWithGrad(Tensor predictions, Tensor targets)
    {
        if (!predictions.Shape.SequenceEqual(targets.Shape))
            throw new ArgumentException("Предсказания и цели должны иметь одинаковые размерности");

        Tensor diff = predictions - targets;
        Tensor squared = diff * diff;

        Tensor loss;
        if (_reductionMean)
        {
            loss = squared.Sum() / predictions.Size;
        }
        else
        {
            loss = squared.Sum();
        }

        // Градиент MSE: dL/dpredictions = 2*(predictions - targets) / N
        Tensor grad = (diff * 2.0) / (_reductionMean ? predictions.Size : 1.0);

        return (loss, grad);
    }
}
