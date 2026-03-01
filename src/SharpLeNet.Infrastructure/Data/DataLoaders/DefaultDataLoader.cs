using SharpLeNet.Core;
using SharpLeNet.Core.Data;

namespace SharpLeNet.Infrastructure.Data.DataLoaders;

/// <summary>
/// Стандартный DataLoader как в PyTorch
/// </summary>
public class DefaultDataLoader : DataLoader
{
    private readonly Random _rnd;
    private readonly bool _dropLast;

    public DefaultDataLoader(Dataset dataset, int batchSize = 32, bool shuffle = true,
                            bool dropLast = false, int? seed = null)
        : base(dataset, batchSize, shuffle)
    {
        _dropLast = dropLast;
        _rnd = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public override IEnumerator<(Tensor features, Tensor labels)> GetEnumerator()
    {
        // Создаем индексы для текущей эпохи
        var indices = Enumerable.Range(0, _dataset.Count).ToArray();
        if (_shuffle)
        {
            ShuffleIndices(indices);
        }

        int totalBatches = _dropLast
            ? indices.Length / _batchSize
            : (int)Math.Ceiling((double)indices.Length / _batchSize);

        for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++)
        {
            int startIdx = batchIdx * _batchSize;
            int endIdx = Math.Min(startIdx + _batchSize, indices.Length);
            int currentBatchSize = endIdx - startIdx;

            // Пропускаем последний неполнный батч если нужно
            if (_dropLast && currentBatchSize < _batchSize)
                break;

            yield return CreateBatch(indices, startIdx, currentBatchSize);
        }
    }

    private (Tensor features, Tensor labels) CreateBatch(int[] indices, int startIdx, int batchSize)
    {
        // Получаем первый пример для определения размеров
        var firstItem = _dataset.GetItem(0);
        int featureSize = firstItem.features.Size;
        int labelSize = firstItem.labels.Size;

        var batchFeaturesData = new double[batchSize * featureSize];
        var batchLabelsData = new double[batchSize * labelSize];

        for (int i = 0; i < batchSize; i++)
        {
            int sampleIdx = indices[startIdx + i];
            var (features, labels) = _dataset.GetItem(sampleIdx);

            Array.Copy(features.Data, 0, batchFeaturesData, i * featureSize, featureSize);
            Array.Copy(labels.Data, 0, batchLabelsData, i * labelSize, labelSize);
        }

        // Форма для батча: [batch, ...]
        var batchFeatureShape = new int[] { batchSize }.Concat(_dataset.FeatureShape).ToArray();
        var batchLabelShape = new int[] { batchSize }.Concat(_dataset.LabelShape).ToArray();

        return (
            new Tensor(batchFeaturesData, batchFeatureShape),
            new Tensor(batchLabelsData, batchLabelShape)
        );
    }

    private void ShuffleIndices(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = _rnd.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }
}
