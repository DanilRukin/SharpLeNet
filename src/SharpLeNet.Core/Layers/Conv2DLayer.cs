using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLeNet.Core.Layers;

/// <summary>
/// Сверточный 2D слой
/// </summary>
public class Conv2DLayer : Layer
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    public int InputChannels => _inputChannels;
    public int OutputChannels => _outputChannels;
    public int KernelSize => _kernelSize;
    public int Stride => _stride;
    public int Padding => _padding;

    /// <summary>
    /// Ядра
    /// </summary>
    /// <remarks>
    /// Форма: [output_channels, input_channels, kernel_h, kernel_w]
    /// </remarks>
    public Tensor Kernels { get; }

    /// <summary>
    /// Смещения
    /// </summary>
    /// <remarks>
    /// Форма: [output_channels]
    /// </remarks>
    public Tensor Biases { get; }

    public Conv2DLayer(int inputChannels, int outputChannels, int kernelSize,
        int stride = 1, int padding = 0)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Инициализация весов
        double stddev = Math.Sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        Random rnd = new();

        // Ядра свертки
        double[] kernelData = new double[outputChannels * inputChannels * kernelSize * kernelSize];
        for (int i = 0; i < kernelData.Length; i++)
        {
            kernelData[i] = rnd.NextDouble() * 2 * stddev - stddev;
        }
        Kernels = new Tensor(kernelData, [outputChannels, inputChannels, kernelSize, kernelSize], true);

        // Смещения
        double[] biasesData = new double[outputChannels];
        for (int i = 0; i < biasesData.Length; i++)
        {
            biasesData[i] = 0.01;
        }
        Biases = new Tensor(biasesData, [outputChannels], true);

        Parameters.Add(Kernels);
        Parameters.Add(Biases);
    }
    public override Tensor Forward(Tensor input)
    {
        // input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            throw new ArgumentException("Для сверточного 2D слоя на входе ожидается " +
                "4D тензор [batch, channels, height, width]!");
        int batchSize = input.Shape[0];
        int inputChannels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];

        if (inputChannels != _inputChannels)
            throw new ArgumentException($"Ожидалось входных каналов {_inputChannels}, " +
                $"но получено {inputChannels}");

        // Вычисляем размеры выхода
        int outputHeight = (inputHeight + 2 * _padding - _kernelSize) / _stride + 1;
        int outputWidth = (inputWidth + 2 * _padding - _kernelSize) / _stride + 1;

        // Создаем выходной тензор
        Tensor output = new([batchSize, _outputChannels, outputHeight, outputWidth]);

        // Добавляем padding если нужно
        Tensor paddedInput = input;
        if (_padding > 0)
        {
            paddedInput = PadTensor(input, _padding);
        }

        // Наивная реализация свертки (для понимания)
        for (int b = 0; b < batchSize; b++)
        {
            for (int oc = 0; oc < _outputChannels; oc++) // по выходным каналам
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        double sum = Biases[oc]; // начинаем со смещения

                        for (int ic = 0; ic < _inputChannels; ic++) // по входным каналам
                        {
                            for (int kh = 0; kh < _kernelSize; kh++) // по высоте ядра
                            {
                                for (int kw = 0; kw < _kernelSize; kw++) // по ширине ядра
                                {
                                    int inputH = oh * _stride + kh;
                                    int inputW = ow * _stride + kw;

                                    double inputVal = paddedInput[b, ic, inputH, inputW];
                                    double kernelVal = Kernels[oc, ic, kh, kw];

                                    sum += inputVal * kernelVal;
                                }
                            }
                        }

                        output[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        return output;
    }

    private Tensor PadTensor(Tensor input, int padding)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int paddedHeight = height + 2 * padding;
        int paddedWidth = width + 2 * padding;

        var padded = new Tensor(new int[] { batchSize, channels, paddedHeight, paddedWidth });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        padded[b, c, h + padding, w + padding] = input[b, c, h, w];
                    }
                }
            }
        }

        return padded;
    }
}
