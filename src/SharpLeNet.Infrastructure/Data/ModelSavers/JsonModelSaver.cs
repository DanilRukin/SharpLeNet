using SharpLeNet.Core;
using SharpLeNet.Core.Data;
using SharpLeNet.Core.Layers;
using SharpLeNet.Infrastructure.Data.ModelSavers.JsonContracts;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpLeNet.Infrastructure.Data.ModelSavers;

/// <summary>
/// JSON сохранятель моделей
/// </summary>
public class JsonModelSaver : IModelSaver
{
    private readonly string _directory;
    private readonly JsonSerializerOptions _jsonOptions;

    public JsonModelSaver(string directory)
    {
        _directory = directory;
        Directory.CreateDirectory(directory);

        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    public void Save(Model model, string identifier)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(identifier))
            throw new ArgumentException("Идентификтор модели не может быть пустым", nameof(identifier));

        string filePath = Path.Combine(_directory, $"{identifier}.json");

        var modelData = new ModelData();
        var layers = GetModelLayers(model);

        foreach (var layer in layers)
        {
            var layerData = new LayerData
            {
                Type = layer.GetType().Name
            };

            // Сохраняем веса и смещения
            foreach (var param in layer.Parameters)
            {
                layerData.Parameters.Add(new TensorData
                {
                    Shape = param.Shape.ToArray(),
                    Data = param.Data.ToArray()
                });
            }

            // Сохраняем специфичные параметры слоя
            switch (layer)
            {
                case LinearLayer linear:
                    layerData.LinearParams = new LinearParams
                    {
                        InputSize = linear.InputSize,
                        OutputSize = linear.OutputSize
                    };
                    break;

                case Conv2DLayer conv:
                    layerData.ConvParams = new ConvParams
                    {
                        InputChannels = conv.InputChannels,
                        OutputChannels = conv.OutputChannels,
                        KernelSize = conv.KernelSize,
                        Stride = conv.Stride,
                        Padding = conv.Padding
                    };
                    break;

                // Слои без параметров ничего не добавляют
                case ReLULayer:
                case SoftmaxLayer:
                case FlattenLayer:
                case MaxPoolingLayer:
                case AvgPoolingLayer:
                    break;

                default:
                    throw new NotSupportedException($"Неподдерживаемый слой: {layer.GetType().Name}");
            }

            modelData.Layers.Add(layerData);
        }

        // Добавляем метаданные
        modelData.Metadata["modelType"] = model.GetType().Name;
        modelData.Metadata["totalLayers"] = layers.Count.ToString();
        modelData.Metadata["totalParameters"] = model.Parameters.Sum(p => p.Size).ToString();

        string json = JsonSerializer.Serialize(modelData, _jsonOptions);
        File.WriteAllText(filePath, json);
    }

    public Model Load(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
            throw new ArgumentException("Идентификатор модели не может быть пустым", nameof(identifier));

        string filePath = Path.Combine(_directory, $"{identifier}.json");
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Модель {identifier} не найдена в {filePath}");

        string json = File.ReadAllText(filePath);
        var modelData = JsonSerializer.Deserialize<ModelData>(json, _jsonOptions);

        if (modelData == null)
            throw new InvalidOperationException("Не удалось десериализовать модель");

        var model = new Model();

        foreach (var layerData in modelData.Layers)
        {
            Layer layer = CreateLayerFromData(layerData);

            // Загружаем веса
            for (int i = 0; i < layer.Parameters.Count; i++)
            {
                if (i >= layerData.Parameters.Count)
                    throw new InvalidOperationException($"Недостаточное кол-во параметров для слоя {layerData.Type}");

                var tensorData = layerData.Parameters[i];

                // Проверяем форму
                if (!layer.Parameters[i].Shape.SequenceEqual(tensorData.Shape))
                    throw new InvalidOperationException(
                        $"Несовпадение размерностей слоя {layerData.Type} параметр {i}. " +
                        $"Ожидалось [{string.Join(", ", layer.Parameters[i].Shape)}], " +
                        $"получено [{string.Join(", ", tensorData.Shape)}]");

                // Проверяем размер данных
                if (layer.Parameters[i].Size != tensorData.Data.Length)
                    throw new InvalidOperationException(
                        $"Несоответствие размера данных для слоя {layerData.Type} параметр {i}. " +
                        $"Ожидалось {layer.Parameters[i].Size}, получено {tensorData.Data.Length}");

                Array.Copy(tensorData.Data, layer.Parameters[i].Data, tensorData.Data.Length);
            }

            model.AddLayer(layer);
        }

        return model;
    }

    public bool Exists(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
            return false;

        string filePath = Path.Combine(_directory, $"{identifier}.json");
        return File.Exists(filePath);
    }

    /// <summary>
    /// Получает список слоев модели через рефлексию
    /// </summary>
    private List<Layer> GetModelLayers(Model model)
    {
        var field = typeof(Model).GetField("_layers",
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Instance);

        if (field == null)
            throw new InvalidOperationException("Не удалось получить доступ к полю _layers в классе модели");

        return field.GetValue(model) as List<Layer> ?? new List<Layer>();
    }

    /// <summary>
    /// Создает слой из данных
    /// </summary>
    private Layer CreateLayerFromData(LayerData layerData)
    {
        return layerData.Type switch
        {
            nameof(LinearLayer) => CreateLinearLayer(layerData),
            nameof(Conv2DLayer) => CreateConvLayer(layerData),
            nameof(ReLULayer) => new ReLULayer(),
            nameof(SoftmaxLayer) => new SoftmaxLayer(),
            nameof(FlattenLayer) => new FlattenLayer(),
            nameof(MaxPoolingLayer) => new MaxPoolingLayer(),
            nameof(AvgPoolingLayer) => new AvgPoolingLayer(),
            _ => throw new NotSupportedException($"Неподдерживаемый слой: {layerData.Type}")
        };
    }

    private LinearLayer CreateLinearLayer(LayerData layerData)
    {
        if (layerData.LinearParams == null)
            throw new InvalidOperationException("Пропущены параметры для линейного слоя");

        var layer = new LinearLayer(
            layerData.LinearParams.InputSize,
            layerData.LinearParams.OutputSize
        );

        return layer;
    }

    private Conv2DLayer CreateConvLayer(LayerData layerData)
    {
        if (layerData.ConvParams == null)
            throw new InvalidOperationException("Пропущены параметры сверточного слоя");

        var layer = new Conv2DLayer(
            layerData.ConvParams.InputChannels,
            layerData.ConvParams.OutputChannels,
            layerData.ConvParams.KernelSize,
            layerData.ConvParams.Stride,
            layerData.ConvParams.Padding
        );

        return layer;
    }

    /// <summary>
    /// Получает список всех файлов сохраненных моделей
    /// </summary>
    public string[] GetSavedModels()
    {
        return Directory.GetFiles(_directory, "*.json")
            .Select(f => Path.GetFileNameWithoutExtension(f))
            .ToArray();
    }

    /// <summary>
    /// Удаляет сохраненную модель
    /// </summary>
    public void Delete(string identifier)
    {
        string filePath = Path.Combine(_directory, $"{identifier}.json");
        if (File.Exists(filePath))
            File.Delete(filePath);
    }

    /// <summary>
    /// Получает метаданные модели без загрузки всей модели
    /// </summary>
    public ModelMetadata GetMetadata(string identifier)
    {
        string filePath = Path.Combine(_directory, $"{identifier}.json");
        if (!File.Exists(filePath))
            return null;

        string json = File.ReadAllText(filePath);
        var modelData = JsonSerializer.Deserialize<ModelData>(json, _jsonOptions);

        return new ModelMetadata
        {
            Identifier = identifier,
            CreatedAt = modelData?.CreatedAt ?? DateTime.MinValue,
            TotalLayers = modelData?.Layers.Count ?? 0,
            Metadata = modelData?.Metadata ?? new Dictionary<string, string>()
        };
    }
}
