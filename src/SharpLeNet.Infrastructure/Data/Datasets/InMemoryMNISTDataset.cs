using SharpLeNet.Core;
using SharpLeNet.Core.Data;

namespace SharpLeNet.Infrastructure.Data.Datasets;

/// <summary>
/// MNIST датасет с загрузкой всех данных в память
/// </summary>
public class InMemoryMNISTDataset : Dataset
{
    private readonly Tensor _features;
    private readonly Tensor _labels;
    private readonly int[] _featureShape;
    private readonly int[] _labelShape;

    public override int Count { get; }
    public override int[] FeatureShape => _featureShape;
    public override int[] LabelShape => _labelShape;

    public InMemoryMNISTDataset(string imagesPath, string labelsPath, bool normalize = true)
    {
        using var imagesStream = File.OpenRead(imagesPath);
        using var imagesReader = new BinaryReader(imagesStream);

        int magicNumber = ReverseBytes(imagesReader.ReadInt32());
        int numSamples = ReverseBytes(imagesReader.ReadInt32());
        int rows = ReverseBytes(imagesReader.ReadInt32());
        int cols = ReverseBytes(imagesReader.ReadInt32());

        if (magicNumber != 2051)
            throw new Exception("Неверный magic number для файла изображений");

        using var labelsStream = File.OpenRead(labelsPath);
        using var labelsReader = new BinaryReader(labelsStream);

        magicNumber = ReverseBytes(labelsReader.ReadInt32());
        int numLabels = ReverseBytes(labelsReader.ReadInt32());

        if (numLabels != numSamples)
            throw new Exception("Количество меток не совпадает");

        Count = numSamples;
        _featureShape = new int[] { 1, rows, cols }; // [C, H, W]
        _labelShape = new int[] { 10 }; // one-hot

        int imageSize = rows * cols;
        var featuresData = new double[numSamples * imageSize];
        var labelsData = new double[numSamples * 10];

        for (int i = 0; i < numSamples; i++)
        {
            // Читаем изображение
            for (int j = 0; j < imageSize; j++)
            {
                byte pixel = imagesReader.ReadByte();
                featuresData[i * imageSize + j] = normalize ? pixel / 255.0 : pixel;
            }

            // Читаем метку
            byte label = labelsReader.ReadByte();
            labelsData[i * 10 + label] = 1.0;
        }

        _features = new Tensor(featuresData, new int[] { numSamples, 1, rows, cols });
        _labels = new Tensor(labelsData, new int[] { numSamples, 10 });
    }

    public override (Tensor features, Tensor labels) GetItem(int index)
    {
        int imageSize = _features.Size / Count;
        int labelSize = _labels.Size / Count;

        var featuresData = new double[imageSize];
        var labelsData = new double[labelSize];

        Array.Copy(_features.Data, index * imageSize, featuresData, 0, imageSize);
        Array.Copy(_labels.Data, index * labelSize, labelsData, 0, labelSize);

        return (
            new Tensor(featuresData, _featureShape),
            new Tensor(labelsData, _labelShape)
        );
    }

    private static int ReverseBytes(int value)
    {
        return (int)((value & 0x000000FFU) << 24 |
                    (value & 0x0000FF00U) << 8 |
                    (value & 0x00FF0000U) >> 8 |
                    (value & 0xFF000000U) >> 24);
    }

    public override void Dispose()
    {
        _features?.Dispose();
        _labels?.Dispose();
    }
}

