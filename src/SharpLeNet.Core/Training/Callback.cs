namespace SharpLeNet.Core.Training;

/// <summary>
/// Базовый класс для callback'ов
/// </summary>
public abstract class Callback
{
    public virtual void OnTrainBegin() { }
    public virtual void OnTrainEnd() { }
    public virtual void OnEpochBegin(int epoch) { }
    public virtual void OnEpochEnd(int epoch, double trainLoss, double valLoss, double trainAcc, double valAcc) { }
    public virtual void OnBatchEnd(int batch, double loss) { }
}
