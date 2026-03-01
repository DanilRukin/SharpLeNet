using SharpLeNet.Core;
using SharpLeNet.Core.Data;

namespace SharpLeNet.Infrastructure.Data.Datasets;

/// <summary>
/// Синтетический датасет для тестирования
/// </summary>
public class RandomDataset : Dataset
{
    private readonly Tensor _features;
    private readonly Tensor _labels;
    private readonly int[] _featureShape;
    private readonly int[] _labelShape;

    public override int Count { get; }
    public override int[] FeatureShape => _featureShape;
    public override int[] LabelShape => _labelShape;

    public RandomDataset(int numSamples, int numFeatures, int numClasses, int? seed = null)
    {
        Count = numSamples;
        _featureShape = new int[] { numFeatures };
        _labelShape = new int[] { numClasses };

        var rnd = seed.HasValue ? new Random(seed.Value) : new Random();

        // Фичи в памяти
        var featuresData = new double[numSamples * numFeatures];
        for (int i = 0; i < featuresData.Length; i++)
            featuresData[i] = rnd.NextDouble() * 2 - 1;

        // One-hot метки
        var labelsData = new double[numSamples * numClasses];
        for (int i = 0; i < numSamples; i++)
            labelsData[i * numClasses + rnd.Next(numClasses)] = 1.0;

        _features = new Tensor(featuresData, new int[] { numSamples, numFeatures });
        _labels = new Tensor(labelsData, new int[] { numSamples, numClasses });
    }

    public override (Tensor features, Tensor labels) GetItem(int index)
    {
        int featureSize = _features.Size / Count;
        int labelSize = _labels.Size / Count;

        var featuresData = new double[featureSize];
        var labelsData = new double[labelSize];

        Array.Copy(_features.Data, index * featureSize, featuresData, 0, featureSize);
        Array.Copy(_labels.Data, index * labelSize, labelsData, 0, labelSize);

        return (
            new Tensor(featuresData, _featureShape),
            new Tensor(labelsData, _labelShape)
        );
    }

    public override void Dispose()
    {
        _features?.Dispose();
        _labels?.Dispose();
    }
}

