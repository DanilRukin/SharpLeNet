namespace SharpLeNet.Vision.Wpf.Models;

public class TrainingProgress
{
    public int CurrentEpoch { get; set; }
    public int TotalEpochs { get; set; }
    public int CurrentBatch { get; set; }
    public int TotalBatches { get; set; }
    public double CurrentLoss { get; set; }
    public double? ValidationLoss { get; set; }
    public double? Accuracy { get; set; }
    public TimeSpan ElapsedTime { get; set; }
    public TimeSpan? EstimatedTimeRemaining { get; set; }
}
