using SharpLeNet.Core;
using SharpLeNet.Core.Data;
using SharpLeNet.Core.Training;

namespace SharpLeNet.Infrastructure.Training.Callbacks;

/// <summary>
/// Сохранение лучшей модели с возможностью выбора формата
/// </summary>
public class ModelCheckpoint : Callback
{
    private readonly Model _model;
    private readonly IModelSaver _modelSaver;
    private readonly string _basePath;
    private double _bestLoss;

    public ModelCheckpoint(Model model, IModelSaver modelSaver, string basePath)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _modelSaver = modelSaver ?? throw new ArgumentNullException(nameof(modelSaver));
        _basePath = basePath;
        _bestLoss = double.MaxValue;
    }

    public override void OnEpochEnd(int epoch, double trainLoss, double valLoss,
                                   double trainAcc, double valAcc)
    {
        if (valLoss < _bestLoss)
        {
            _bestLoss = valLoss;
            string identifier = $"epoch_{epoch + 1}_loss_{valLoss:F6}";

            _modelSaver.Save(_model, identifier);
            Console.WriteLine($"  ✓ Модель сохранена (лучшая val_loss: {valLoss:F6})");
        }
    }
}
