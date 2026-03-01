using System.Collections;
using System.Data;

namespace SharpLeNet.Core.Data;

/// <summary>
/// Базовый класс для всех датасетов
/// </summary>
public abstract class Dataset : IDisposable
{
    /// <summary>
    /// Количество примеров в датасете
    /// </summary>
    public abstract int Count { get; }

    /// <summary>
    /// Возвращает один пример по индексу
    /// </summary>
    public abstract (Tensor features, Tensor labels) GetItem(int index);

    /// <summary>
    /// Форма признаков для одного примера
    /// </summary>
    public abstract int[] FeatureShape { get; }

    /// <summary>
    /// Форма меток для одного примера
    /// </summary>
    public abstract int[] LabelShape { get; }

    public virtual void Dispose() { }
}

