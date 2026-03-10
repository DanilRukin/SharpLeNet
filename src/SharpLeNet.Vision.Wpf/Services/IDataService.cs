using SharpLeNet.Core;
using SharpLeNet.Vision.Wpf.Models;

namespace SharpLeNet.Vision.Wpf.Services;

public interface IDataService
{
    bool IsLoaded { get; }
    int TrainingImageCount { get; }
    int TestImageCount { get; }

    Task<bool> LoadMNISTAsync(IProgress<double>? progress = null);
    Tensor GetTrainingImages();
    Tensor GetTrainingLabels();
    Tensor GetTestImages();
    Tensor GetTestLabels();
    MNISTSample GetSample(int index, bool isTraining = true);
    List<MNISTSample> GetPreviewSamples(int count = 20, bool isTraining = true);
}
