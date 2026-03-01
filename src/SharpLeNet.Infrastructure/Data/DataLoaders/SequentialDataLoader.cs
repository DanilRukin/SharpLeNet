using SharpLeNet.Core;
using SharpLeNet.Core.Data;

namespace SharpLeNet.Infrastructure.Data.DataLoaders;

/// <summary>
/// DataLoader для последовательной загрузки (без перемешивания)
/// </summary>
public class SequentialDataLoader : DataLoader
{
    public SequentialDataLoader(Dataset dataset, int batchSize = 32)
        : base(dataset, batchSize, shuffle: false) { }

    public override IEnumerator<(Tensor features, Tensor labels)> GetEnumerator()
    {
        for (int startIdx = 0; startIdx < _dataset.Count; startIdx += _batchSize)
        {
            int endIdx = Math.Min(startIdx + _batchSize, _dataset.Count);
            int batchSize = endIdx - startIdx;

            yield return CreateBatch(startIdx, batchSize);
        }
    }

    private (Tensor features, Tensor labels) CreateBatch(int startIdx, int batchSize)
    {
        var firstItem = _dataset.GetItem(0);
        int featureSize = firstItem.features.Size;
        int labelSize = firstItem.labels.Size;

        var batchFeaturesData = new double[batchSize * featureSize];
        var batchLabelsData = new double[batchSize * labelSize];

        for (int i = 0; i < batchSize; i++)
        {
            var (features, labels) = _dataset.GetItem(startIdx + i);

            Array.Copy(features.Data, 0, batchFeaturesData, i * featureSize, featureSize);
            Array.Copy(labels.Data, 0, batchLabelsData, i * labelSize, labelSize);
        }

        var batchFeatureShape = new int[] { batchSize }.Concat(_dataset.FeatureShape).ToArray();
        var batchLabelShape = new int[] { batchSize }.Concat(_dataset.LabelShape).ToArray();

        return (
            new Tensor(batchFeaturesData, batchFeatureShape),
            new Tensor(batchLabelsData, batchLabelShape)
        );
    }
}
