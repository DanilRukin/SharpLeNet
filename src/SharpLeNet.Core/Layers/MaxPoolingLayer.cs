namespace SharpLeNet.Core.Layers;

/// <summary>
/// Слой MaxPooling
/// </summary>
public class MaxPoolingLayer : Layer
{
    private readonly int _poolSize;
    private readonly int _stride;

    // Для backward нужно запоминать индексы максимумов
    private int[]? _maxIndices;

    public MaxPoolingLayer(int poolSize = 2, int stride = 2)
    {
        _poolSize = poolSize;
        _stride = stride;
    }

    public override Tensor Forward(Tensor input)
    {
        if (input.Rank != 4)
            throw new ArgumentException("MaxPooling слой ожидает 4D тензор на вход");

        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height - _poolSize) / _stride + 1;
        int outputWidth = (width - _poolSize) / _stride + 1;

        var output = new Tensor(new int[] { batchSize, channels, outputHeight, outputWidth });
        _maxIndices = new int[output.Size]; // запоминаем индексы для backward

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Находим максимум в окне poolSize x poolSize
                        double maxVal = double.MinValue;
                        int maxIndex = -1;

                        for (int ph = 0; ph < _poolSize; ph++)
                        {
                            for (int pw = 0; pw < _poolSize; pw++)
                            {
                                int inputH = oh * _stride + ph;
                                int inputW = ow * _stride + pw;

                                double val = input[b, c, inputH, inputW];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIndex = inputH * width + inputW; // линейный индекс во входе
                                }
                            }
                        }

                        int outputIndex = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        output.Data[outputIndex] = maxVal;
                        _maxIndices[outputIndex] = maxIndex;
                    }
                }
            }
        }

        return output;
    }
}
