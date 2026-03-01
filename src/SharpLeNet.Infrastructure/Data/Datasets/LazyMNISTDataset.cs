using SharpLeNet.Core;
using SharpLeNet.Core.Data;

namespace SharpLeNet.Infrastructure.Data.Datasets;

/// <summary>
/// MNIST датасет с ленивой загрузкой - не хранит все данные в памяти!
/// </summary>
public class LazyMNISTDataset : Dataset
{
    private readonly string _imagesPath;
    private readonly string _labelsPath;
    private readonly bool _normalize;
    private readonly int _numSamples;
    private readonly int _rows;
    private readonly int _cols;
    private readonly int[] _featureShape;
    private readonly int[] _labelShape;

    private FileStream? _imagesStream;
    private BinaryReader? _imagesReader;
    private FileStream? _labelsStream;
    private BinaryReader? _labelsReader;

    public override int Count => _numSamples;
    public override int[] FeatureShape => _featureShape;
    public override int[] LabelShape => _labelShape;

    public LazyMNISTDataset(string imagesPath, string labelsPath, bool normalize = true)
    {
        _imagesPath = imagesPath;
        _labelsPath = labelsPath;
        _normalize = normalize;

        // Открываем файлы и читаем только заголовки
        _imagesStream = File.OpenRead(imagesPath);
        _imagesReader = new BinaryReader(_imagesStream);

        int magicNumber = ReverseBytes(_imagesReader.ReadInt32());
        _numSamples = ReverseBytes(_imagesReader.ReadInt32());
        _rows = ReverseBytes(_imagesReader.ReadInt32());
        _cols = ReverseBytes(_imagesReader.ReadInt32());

        if (magicNumber != 2051)
            throw new Exception("Неверный magic number");

        _labelsStream = File.OpenRead(labelsPath);
        _labelsReader = new BinaryReader(_labelsStream);

        magicNumber = ReverseBytes(_labelsReader.ReadInt32());
        int numLabels = ReverseBytes(_labelsReader.ReadInt32());

        if (numLabels != _numSamples)
            throw new Exception("Количество меток не совпадает");

        _featureShape = new int[] { 1, _rows, _cols };
        _labelShape = new int[] { 10 };
    }

    public override (Tensor features, Tensor labels) GetItem(int index)
    {
        lock (this) // Потокобезопасность для параллельной загрузки
        {
            // Позиция изображения: заголовок (16 байт) + индекс * размер_изображения
            int imageSize = _rows * _cols;
            long imagePosition = 16 + index * imageSize;
            _imagesStream!.Seek(imagePosition, SeekOrigin.Begin);

            var imageData = new byte[imageSize];
            _imagesStream.Read(imageData, 0, imageSize);

            // Позиция метки: заголовок (8 байт) + индекс
            long labelPosition = 8 + index;
            _labelsStream!.Seek(labelPosition, SeekOrigin.Begin);
            int label = _labelsStream.ReadByte();
            label = label == -1 ? 0 : label;

            // Конвертируем в double
            var featuresData = new double[imageSize];
            for (int i = 0; i < imageSize; i++)
            {
                featuresData[i] = _normalize ? imageData[i] / 255.0 : imageData[i];
            }

            var labelsData = new double[10];
            labelsData[label] = 1.0;

            return (
                new Tensor(featuresData, _featureShape),
                new Tensor(labelsData, _labelShape)
            );
        }
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
        _imagesReader?.Dispose();
        _imagesStream?.Dispose();
        _labelsReader?.Dispose();
        _labelsStream?.Dispose();
    }
}

