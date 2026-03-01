using SharpLeNet.Core.Data;

namespace SharpLeNet.Core.Training;

/// <summary>
/// Базовый callback для сохранения модели
/// </summary>
public abstract class ModelCheckpointBase : Callback
{
    protected readonly IModelSaver _modelSaver;
    protected readonly string _basePath;
    protected double _bestLoss;

    protected ModelCheckpointBase(IModelSaver modelSaver, string basePath)
    {
        _modelSaver = modelSaver ?? throw new ArgumentNullException(nameof(modelSaver));
        _basePath = basePath;
        _bestLoss = double.MaxValue;
    }
}
