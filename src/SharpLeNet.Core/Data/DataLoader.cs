namespace SharpLeNet.Core.Data;

/// <summary>
/// Базовый класс для DataLoader'ов
/// </summary>
public abstract class DataLoader : IEnumerable<(Tensor features, Tensor labels)>, IDisposable
{
    protected readonly Dataset _dataset;
    protected readonly int _batchSize;
    protected readonly bool _shuffle;

    public DataLoader(Dataset dataset, int batchSize = 32, bool shuffle = true)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentException("Batch size must be positive");
        _shuffle = shuffle;
    }

    public abstract IEnumerator<(Tensor features, Tensor labels)> GetEnumerator();

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public virtual void Dispose() => _dataset?.Dispose();
}
