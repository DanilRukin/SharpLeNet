using SharpLeNet.Core;
using SharpLeNet.Core.Data;
using SharpLeNet.Core.Layers;

namespace SharpLeNet.Infrastructure.Data.ModelSavers;

/// <summary>
/// Бинарный сохранятель моделей
/// </summary>
public class BinaryModelSaver : IModelSaver
{
    private readonly string _directory;

    public BinaryModelSaver(string directory)
    {
        _directory = directory;
        Directory.CreateDirectory(directory);
    }

    public void Save(Model model, string identifier)
    {
        string filePath = Path.Combine(_directory, $"{identifier}.bin");

        using var stream = File.Create(filePath);
        using var writer = new BinaryWriter(stream);

        // Получаем слои модели через рефлексию
        var layers = GetModelLayers(model);

        // Сохраняем количество слоев
        writer.Write(layers.Count);

        foreach (var layer in layers)
        {
            // Сохраняем тип слоя
            writer.Write(layer.GetType().FullName ?? "");

            // Сохраняем параметры слоя (веса, смещения)
            SaveLayer(layer, writer);

            // Сохраняем веса
            foreach (var param in layer.Parameters)
            {
                SaveTensor(param, writer);
            }
        }
    }

    public Model Load(string identifier)
    {
        string filePath = Path.Combine(_directory, $"{identifier}.bin");
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Модель {identifier} не найдена");

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        int layerCount = reader.ReadInt32();
        var model = new Model();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            var layer = CreateLayer(layerTypeName, reader);

            // Загружаем веса
            foreach (var param in layer.Parameters)
            {
                LoadTensor(param, reader);
            }

            model.AddLayer(layer);
        }

        return model;
    }

    public bool Exists(string identifier)
    {
        return File.Exists(Path.Combine(_directory, $"{identifier}.bin"));
    }

    private List<Layer> GetModelLayers(Model model)
    {
        var field = typeof(Model).GetField("_layers",
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Instance);

        return field?.GetValue(model) as List<Layer> ?? new List<Layer>();
    }

    private void SaveLayer(Layer layer, BinaryWriter writer)
    {
        switch (layer)
        {
            case LinearLayer linear:
                writer.Write(linear.InputSize);
                writer.Write(linear.OutputSize);
                break;

            case Conv2DLayer conv:
                writer.Write(conv.InputChannels);
                writer.Write(conv.OutputChannels);
                writer.Write(conv.KernelSize);
                writer.Write(conv.Stride);
                writer.Write(conv.Padding);
                break;

            // Для слоев без параметров ничего не сохраняем
            case ReLULayer:
            case SoftmaxLayer:
            case FlattenLayer:
            case MaxPoolingLayer:
            case AvgPoolingLayer:
                break;

            default:
                throw new Exception($"Неизвестный тип слоя: {layer.GetType()}");
        }
    }

    private Layer CreateLayer(string typeName, BinaryReader reader)
    {
        return typeName switch
        {
            nameof(LinearLayer) => new LinearLayer(
                reader.ReadInt32(), // inputSize
                reader.ReadInt32()  // outputSize
            ),

            nameof(Conv2DLayer) => new Conv2DLayer(
                reader.ReadInt32(), // inputChannels
                reader.ReadInt32(), // outputChannels
                reader.ReadInt32(), // kernelSize
                reader.ReadInt32(), // stride
                reader.ReadInt32()  // padding
            ),

            nameof(ReLULayer) => new ReLULayer(),
            nameof(SoftmaxLayer) => new SoftmaxLayer(),
            nameof(FlattenLayer) => new FlattenLayer(),
            nameof(MaxPoolingLayer) => new MaxPoolingLayer(),
            nameof(AvgPoolingLayer) => new AvgPoolingLayer(),

            _ => throw new Exception($"Неизвестный тип слоя: {typeName}")
        };
    }

    private void SaveTensor(Tensor tensor, BinaryWriter writer)
    {
        // Сохраняем размерность
        writer.Write(tensor.Rank);
        foreach (int dim in tensor.Shape)
            writer.Write(dim);

        // Сохраняем данные
        foreach (double value in tensor.Data)
            writer.Write(value);
    }

    private void LoadTensor(Tensor tensor, BinaryReader reader)
    {
        // Проверяем размерность
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();

        if (!tensor.Shape.SequenceEqual(shape))
            throw new Exception("Несовпадение формы тензора");

        // Загружаем данные
        for (int i = 0; i < tensor.Size; i++)
            tensor.Data[i] = reader.ReadDouble();
    }
}
