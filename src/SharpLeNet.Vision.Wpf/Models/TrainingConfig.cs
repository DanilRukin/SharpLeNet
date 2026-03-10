namespace SharpLeNet.Vision.Wpf.Models;

public class TrainingConfig
{
    public NetworkModels ModelType { get; set; } = NetworkModels.LeNet5;
    public int Epochs { get; set; } = 5;
    public int BatchSize { get; set; } = 32;
    public double LearningRate { get; set; } = 0.001;
    public NetworkOptimizers Optimizer { get; set; } = NetworkOptimizers.Adam;
}
