using Microsoft.Extensions.Logging;

namespace SharpLeNet.Core.Training;

/// <summary>
/// Ранняя остановка при отсутствии улучшений
/// </summary>
public class EarlyStopping : Callback
{
    private readonly int _patience;
    private readonly double _minDelta;
    private int _wait;
    private double _bestLoss;
    private bool _stopTraining;

    public EarlyStopping(ILogger<EarlyStopping> logger, int patience = 5, double minDelate = 1e-4)
    {
        _patience = patience;
        _minDelta = minDelate;
        _wait = 0;
        _stopTraining = false;
        _bestLoss = double.MaxValue;
    }

    public override void OnEpochEnd(int epoch, double trainLoss, double valLoss, double trainAcc, double valAcc)
    {
        if (valLoss < _bestLoss - _minDelta)
        {
            _bestLoss = valLoss;
            _wait = 0;
        }
        else
        {
            _wait++;
            if (_wait > _patience)
            {
                Console.WriteLine($"\nРанняя остановка на эпохе {epoch + 1}");
                _stopTraining = true;
            }
        }
    }

    public bool ShouldStop => _stopTraining;
}
