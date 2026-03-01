namespace SharpLeNet.Core.Layers;

/// <summary>
/// Полносвязный слой
/// </summary>
public class LinearLayer : Layer
{
    private readonly int _inputSize;
    private readonly int _outputSize;

    public int InputSize => _outputSize;

    public int OutputSize => _outputSize;

    /// <summary>
    /// Веса
    /// </summary>
    public Tensor Weights { get; }

    /// <summary>
    /// Смещения
    /// </summary>
    public Tensor Biases { get; }

    public LinearLayer(int inputSize, int outputSize)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;

        // Инициализация весов (Xavier/Glorot)
        double stddev = Math.Sqrt(2.0 / (_inputSize + _outputSize));
        Random rnd = new();

        // Веса
        double[] weightsData = new double[_inputSize * _outputSize];
        for (int i = 0; i < weightsData.Length; i++)
        {
            weightsData[i] = rnd.NextDouble() * 2 * stddev - stddev; // равномерное распределение
        }
        Weights = new(weightsData, [_inputSize, _outputSize], true);

        // Смещения (инициализируем нулями или маленькими значениями)
        double[] biasesData = new double[_outputSize];
        for (int i = 0; i < _outputSize; i++)
        {
            biasesData[i] = 0.01;
        }
        Biases = new(biasesData, [_outputSize], true);

        // Добавляем параметры в список
        Parameters.Add(Weights);
        Parameters.Add(Biases);
    }

    public override Tensor Forward(Tensor input)
    {
        if (input.Rank != 2)
            throw new ArgumentException("Ожидался двумерный тензор [batch_size, input_size]");
        if (input.Shape[1] != _inputSize)
            throw new ArgumentException($"Ожидался входной размер {_inputSize}. " +
                $"Но получено {input.Shape[1]}");
        int batchSize = input.Shape[0];

        // Вычисляем: output = input * W + b
        // input: [batch_size, input_size]
        // W: [input_size, output_size]
        // b: [output_size]

        Tensor wTransposed = Weights; // [input_size, output_size]
        Tensor output = input.MatMul(wTransposed); // [batch_size, output_size]

        // Добавляем смещение (broadcast по batch dimension)
        Tensor broadcastedBias = BroadcastBias(Biases, batchSize);
        output = output + broadcastedBias;

        return output;
    }

    private Tensor BroadcastBias(Tensor bias, int batchSize)
    {
        return bias.Broadcast([batchSize, bias.Size]);
    }
}
