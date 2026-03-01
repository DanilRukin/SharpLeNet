using SharpLeNet.Core;
using SharpLeNet.Core.Data;
using SharpLeNet.Core.Layers;
using SharpLeNet.Infrastructure.Data.ModelSavers.DbContracts;
using SharpLeNet.Infrastructure.Data.SqlConnections;
using System.Data;
using System.Text.Json;

namespace SharpLeNet.Infrastructure.Data.ModelSavers;

/// <summary>
/// Сохранятель моделей в реляционную БД
/// </summary>
public class DatabaseModelSaver : IModelSaver
{
    private readonly IDatabaseConnection _connection;
    private readonly JsonSerializerOptions _jsonOptions;

    public DatabaseModelSaver(IDatabaseConnection connection)
    {
        _connection = connection ?? throw new ArgumentNullException(nameof(connection));
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        InitializeDatabase();
    }

    /// <summary>
    /// Создает таблицы если их нет
    /// </summary>
    private void InitializeDatabase()
    {
        _connection.Open();

        try
        {
            // Таблица моделей
            _connection.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS Models (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Identifier TEXT UNIQUE NOT NULL,
                        CreatedAt DATETIME NOT NULL,
                        Description TEXT,
                        Version INTEGER NOT NULL DEFAULT 1,
                        TotalLayers INTEGER NOT NULL DEFAULT 0,
                        TotalParameters INTEGER NOT NULL DEFAULT 0
                    )");

            // Таблица слоев
            _connection.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS Layers (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ModelId INTEGER NOT NULL,
                        LayerIndex INTEGER NOT NULL,
                        LayerType TEXT NOT NULL,
                        
                        -- Параметры линейного слоя
                        InputSize INTEGER,
                        OutputSize INTEGER,
                        
                        -- Параметры сверточного слоя
                        InputChannels INTEGER,
                        OutputChannels INTEGER,
                        KernelSize INTEGER,
                        Stride INTEGER,
                        Padding INTEGER,
                        
                        FOREIGN KEY(ModelId) REFERENCES Models(Id) ON DELETE CASCADE,
                        UNIQUE(ModelId, LayerIndex)
                    )");

            // Таблица тензоров
            _connection.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS Tensors (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        LayerId INTEGER NOT NULL,
                        ParameterIndex INTEGER NOT NULL,
                        ShapeJson TEXT NOT NULL,
                        DataBlob BLOB NOT NULL,
                        DataLength INTEGER NOT NULL,
                        FOREIGN KEY(LayerId) REFERENCES Layers(Id) ON DELETE CASCADE,
                        UNIQUE(LayerId, ParameterIndex)
                    )");

            // Индексы для быстрого поиска
            _connection.ExecuteNonQuery(
                "CREATE INDEX IF NOT EXISTS idx_models_identifier ON Models(Identifier)");
            _connection.ExecuteNonQuery(
                "CREATE INDEX IF NOT EXISTS idx_layers_modelid ON Layers(ModelId)");
            _connection.ExecuteNonQuery(
                "CREATE INDEX IF NOT EXISTS idx_tensors_layerid ON Tensors(LayerId)");
        }
        finally
        {
            _connection.Close();
        }
    }

    public void Save(Model model, string identifier)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(identifier))
            throw new ArgumentException("Идентификатор модели не может быть пустым", nameof(identifier));

        _connection.Open();
        _connection.BeginTransaction();

        try
        {
            // Удаляем старую версию если есть
            if (Exists(identifier))
            {
                DeleteExistingModel(identifier);
            }

            // Получаем слои модели
            var layers = GetModelLayers(model);
            int totalParameters = model.Parameters.Sum(p => p.Size);

            // Сохраняем метаданные модели
            int modelId = SaveModelMetadata(identifier, layers.Count, totalParameters);

            // Сохраняем слои и их тензоры
            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                int layerId = SaveLayer(modelId, layer, i);

                // Сохраняем тензоры (веса, смещения)
                for (int j = 0; j < layer.Parameters.Count; j++)
                {
                    SaveTensor(layerId, layer.Parameters[j], j);
                }
            }

            _connection.CommitTransaction();
        }
        catch
        {
            _connection.RollbackTransaction();
            throw;
        }
        finally
        {
            _connection.Close();
        }
    }

    public Model Load(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
            throw new ArgumentException("Идентификатор модели не может быть пустым", nameof(identifier));

        _connection.Open();

        try
        {
            // Загружаем модель
            var modelData = LoadModelMetadata(identifier);
            if (modelData == null)
                throw new InvalidOperationException($"Модель {identifier} не найдена");

            var model = new Model();

            // Загружаем слои по порядку
            var layers = LoadLayers(modelData.Id);
            foreach (var layerData in layers.OrderBy(l => l.LayerIndex))
            {
                var layer = CreateLayerFromDb(layerData);

                // Загружаем тензоры для слоя
                var tensors = LoadTensors(layerData.Id);
                foreach (var tensorData in tensors.OrderBy(t => t.ParameterIndex))
                {
                    var tensor = layer.Parameters[tensorData.ParameterIndex];
                    LoadTensorData(tensor, tensorData);
                }

                model.AddLayer(layer);
            }

            return model;
        }
        finally
        {
            _connection.Close();
        }
    }

    public bool Exists(string identifier)
    {
        _connection.Open();
        try
        {
            var count = _connection.ExecuteScalar<long>(
                "SELECT COUNT(*) FROM Models WHERE Identifier = @id",
                ("id", identifier));
            return count > 0;
        }
        finally
        {
            _connection.Close();
        }
    }

    /// <summary>
    /// Удаляет существующую модель
    /// </summary>
    private void DeleteExistingModel(string identifier)
    {
        _connection.ExecuteNonQuery(
            "DELETE FROM Models WHERE Identifier = @id",
            ("id", identifier));
    }

    /// <summary>
    /// Сохраняет метаданные модели
    /// </summary>
    private int SaveModelMetadata(string identifier, int totalLayers, int totalParameters)
    {
        _connection.ExecuteNonQuery(@"
                INSERT INTO Models (Identifier, CreatedAt, Version, TotalLayers, TotalParameters)
                VALUES (@id, @createdAt, @version, @totalLayers, @totalParameters)",
            ("id", identifier),
            ("createdAt", DateTime.UtcNow),
            ("version", 1),
            ("totalLayers", totalLayers),
            ("totalParameters", totalParameters));

        return (int)_connection.ExecuteScalar<long>("SELECT last_insert_rowid()");
    }

    /// <summary>
    /// Сохраняет слой
    /// </summary>
    private int SaveLayer(int modelId, Layer layer, int index)
    {
        string sql = @"
                INSERT INTO Layers (
                    ModelId, LayerIndex, LayerType,
                    InputSize, OutputSize,
                    InputChannels, OutputChannels, KernelSize, Stride, Padding
                ) VALUES (
                    @modelId, @index, @type,
                    @inputSize, @outputSize,
                    @inputChannels, @outputChannels, @kernelSize, @stride, @padding
                )";

        var parameters = new List<(string, object)>
            {
                ("modelId", modelId),
                ("index", index),
                ("type", layer.GetType().Name)
            };

        // Добавляем специфичные параметры
        switch (layer)
        {
            case LinearLayer linear:
                parameters.Add(("inputSize", linear.InputSize));
                parameters.Add(("outputSize", linear.OutputSize));
                parameters.Add(("inputChannels", DBNull.Value));
                parameters.Add(("outputChannels", DBNull.Value));
                parameters.Add(("kernelSize", DBNull.Value));
                parameters.Add(("stride", DBNull.Value));
                parameters.Add(("padding", DBNull.Value));
                break;

            case Conv2DLayer conv:
                parameters.Add(("inputSize", DBNull.Value));
                parameters.Add(("outputSize", DBNull.Value));
                parameters.Add(("inputChannels", conv.InputChannels));
                parameters.Add(("outputChannels", conv.OutputChannels));
                parameters.Add(("kernelSize", conv.KernelSize));
                parameters.Add(("stride", conv.Stride));
                parameters.Add(("padding", conv.Padding));
                break;

            default:
                parameters.Add(("inputSize", DBNull.Value));
                parameters.Add(("outputSize", DBNull.Value));
                parameters.Add(("inputChannels", DBNull.Value));
                parameters.Add(("outputChannels", DBNull.Value));
                parameters.Add(("kernelSize", DBNull.Value));
                parameters.Add(("stride", DBNull.Value));
                parameters.Add(("padding", DBNull.Value));
                break;
        }

        _connection.ExecuteNonQuery(sql, parameters.ToArray());
        return (int)_connection.ExecuteScalar<long>("SELECT last_insert_rowid()");
    }

    /// <summary>
    /// Сохраняет тензор
    /// </summary>
    private void SaveTensor(int layerId, Tensor tensor, int parameterIndex)
    {
        string shapeJson = JsonSerializer.Serialize(tensor.Shape, _jsonOptions);
        byte[] dataBlob = SerializeDoubles(tensor.Data);

        _connection.ExecuteNonQuery(@"
                INSERT INTO Tensors (LayerId, ParameterIndex, ShapeJson, DataBlob, DataLength)
                VALUES (@layerId, @paramIndex, @shapeJson, @dataBlob, @dataLength)",
            ("layerId", layerId),
            ("paramIndex", parameterIndex),
            ("shapeJson", shapeJson),
            ("dataBlob", dataBlob),
            ("dataLength", tensor.Size));
    }

    /// <summary>
    /// Загружает метаданные модели
    /// </summary>
    private DbModel? LoadModelMetadata(string identifier)
    {
        return _connection.ExecuteQuery(@"
                SELECT Id, Identifier, CreatedAt, Description, Version, TotalLayers, TotalParameters
                FROM Models WHERE Identifier = @id",
            reader => new DbModel
            {
                Id = reader.GetInt32(0),
                Identifier = reader.GetString(1),
                CreatedAt = reader.GetDateTime(2),
                Description = reader.IsDBNull(3) ? null : reader.GetString(3),
                Version = reader.GetInt32(4),
                TotalLayers = reader.GetInt32(5),
                TotalParameters = reader.GetInt32(6)
            },
            ("id", identifier)).FirstOrDefault();
    }

    /// <summary>
    /// Загружает слои модели
    /// </summary>
    private List<DbLayer> LoadLayers(int modelId)
    {
        return _connection.ExecuteQuery(@"
                SELECT Id, LayerIndex, LayerType,
                       InputSize, OutputSize,
                       InputChannels, OutputChannels, KernelSize, Stride, Padding
                FROM Layers WHERE ModelId = @modelId
                ORDER BY LayerIndex",
            reader => new DbLayer
            {
                Id = reader.GetInt32(0),
                LayerIndex = reader.GetInt32(1),
                LayerType = reader.GetString(2),
                InputSize = reader.IsDBNull(3) ? null : reader.GetInt32(3),
                OutputSize = reader.IsDBNull(4) ? null : reader.GetInt32(4),
                InputChannels = reader.IsDBNull(5) ? null : reader.GetInt32(5),
                OutputChannels = reader.IsDBNull(6) ? null : reader.GetInt32(6),
                KernelSize = reader.IsDBNull(7) ? null : reader.GetInt32(7),
                Stride = reader.IsDBNull(8) ? null : reader.GetInt32(8),
                Padding = reader.IsDBNull(9) ? null : reader.GetInt32(9)
            },
            ("modelId", modelId)).ToList();
    }

    /// <summary>
    /// Загружает тензоры слоя
    /// </summary>
    private List<DbTensor> LoadTensors(int layerId)
    {
        return _connection.ExecuteQuery(@"
                SELECT Id, ParameterIndex, ShapeJson, DataBlob, DataLength
                FROM Tensors WHERE LayerId = @layerId
                ORDER BY ParameterIndex",
            reader => new DbTensor
            {
                Id = reader.GetInt32(0),
                ParameterIndex = reader.GetInt32(1),
                ShapeJson = reader.GetString(2),
                DataBlob = (byte[])reader.GetValue(3),
                DataLength = reader.GetInt32(4)
            },
            ("layerId", layerId)).ToList();
    }

    /// <summary>
    /// Создает слой из данных БД
    /// </summary>
    private Layer CreateLayerFromDb(DbLayer layerData)
    {
        return layerData.LayerType switch
        {
            nameof(LinearLayer) => new LinearLayer(
                layerData.InputSize!.Value,
                layerData.OutputSize!.Value
            ),

            nameof(Conv2DLayer) => new Conv2DLayer(
                layerData.InputChannels!.Value,
                layerData.OutputChannels!.Value,
                layerData.KernelSize!.Value,
                layerData.Stride ?? 1,
                layerData.Padding ?? 0
            ),

            nameof(ReLULayer) => new ReLULayer(),
            nameof(SoftmaxLayer) => new SoftmaxLayer(),
            nameof(FlattenLayer) => new FlattenLayer(),
            nameof(MaxPoolingLayer) => new MaxPoolingLayer(),
            nameof(AvgPoolingLayer) => new AvgPoolingLayer(),

            _ => throw new NotSupportedException($"Неподдерживаемый слой: {layerData.LayerType}")
        };
    }

    /// <summary>
    /// Загружает данные тензора
    /// </summary>
    private void LoadTensorData(Tensor tensor, DbTensor tensorData)
    {
        // Проверяем форму
        var shape = JsonSerializer.Deserialize<int[]>(tensorData.ShapeJson, _jsonOptions);
        if (shape == null || !tensor.Shape.SequenceEqual(shape))
            throw new InvalidOperationException("Несовпадение размерностей");

        // Загружаем данные
        var data = DeserializeDoubles(tensorData.DataBlob, tensorData.DataLength);
        Array.Copy(data, tensor.Data, data.Length);
    }

    /// <summary>
    /// Сериализует массив double в byte[]
    /// </summary>
    private byte[] SerializeDoubles(double[] data)
    {
        byte[] bytes = new byte[data.Length * sizeof(double)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    /// <summary>
    /// Десериализует byte[] в массив double
    /// </summary>
    private double[] DeserializeDoubles(byte[] bytes, int expectedLength)
    {
        double[] data = new double[expectedLength];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
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
            throw new InvalidOperationException("Не удалось получить доступ к полю _layers класса модели");

        return field.GetValue(model) as List<Layer> ?? new List<Layer>();
    }
}
