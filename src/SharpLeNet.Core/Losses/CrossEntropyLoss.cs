namespace SharpLeNet.Core.Losses;

/// <summary>
/// Cross Entropy Loss (сочетается с Softmax)
/// </summary>
public class CrossEntropyLoss : Loss
{
    private readonly bool _reductionMean; // true = mean, false = sum

    public CrossEntropyLoss(bool reductionMean = true)
    {
        _reductionMean = reductionMean;
    }

    public override Tensor Compute(Tensor predictions, Tensor targets)
    {
        var (_, loss) = predictions.SoftmaxCrossEntropy(targets);
        return loss;
    }

    public override (Tensor loss, Tensor grad) ComputeWithGrad(Tensor predictions, Tensor targets)
    {
        var (softmax, loss) = predictions.SoftmaxCrossEntropy(targets);
        // Градиент уже вычислен в backward графе
        // Но вернем явно: dL/dpredictions = softmax - targets
        int batchSize = predictions.Shape[0];
        int numClasses = predictions.Shape[1];

        Tensor grad = new(predictions.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < numClasses; j++)
            {
                grad[i, j] = (softmax[i, j] - targets[i, j]) / (_reductionMean ? batchSize : 1.0);
            }
        }
        return (loss, grad);
    }
}
