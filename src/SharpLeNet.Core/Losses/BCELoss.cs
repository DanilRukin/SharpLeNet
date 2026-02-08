namespace SharpLeNet.Core.Losses;

/// <summary>
/// Binary Cross Entropy Loss (для бинарной классификации)
/// </summary>
public class BCELoss : Loss
{
    private readonly bool _reductionMean;
    private readonly double _eps = 1e-12;

    public BCELoss(bool reductionMean = true)
    {
        _reductionMean = reductionMean;
    }

    public override Tensor Compute(Tensor predictions, Tensor targets)
    {
        if (!predictions.Shape.SequenceEqual(targets.Shape))
            throw new ArgumentException("Предсказания и цели должны иметь одинаковые размерности");
        // BCE = -[y*log(p) + (1-y)*log(1-p)]
        // Для численной стабильности: ограничиваем p в [eps, 1-eps]

        int batchSize = predictions.Shape[0];
        int features = predictions.Shape[1];

        double lossValue = 0.0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < features; j++)
            {
                double p = Math.Max(_eps, Math.Min(1 - _eps, predictions[i, j]));
                double y = targets[i, j];

                lossValue += -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
            }
        }

        if (_reductionMean)
        {
            lossValue /= (batchSize * features);
        }

        return new([lossValue], [1]);
    }

    public override (Tensor loss, Tensor grad) ComputeWithGrad(Tensor predictions, Tensor targets)
    {
        var loss = Compute(predictions, targets);

        // Градиент BCE: dL/dp = (p - y) / (p*(1-p))
        int batchSize = predictions.Shape[0];
        int features = predictions.Shape[1];

        var grad = new Tensor(predictions.Shape);
        double scale = _reductionMean ? 1.0 / (batchSize * features) : 1.0;

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < features; j++)
            {
                double p = Math.Max(_eps, Math.Min(1 - _eps, predictions[i, j]));
                double y = targets[i, j];

                grad[i, j] = (p - y) / (p * (1 - p)) * scale;
            }
        }

        return (loss, grad);
    }
}
