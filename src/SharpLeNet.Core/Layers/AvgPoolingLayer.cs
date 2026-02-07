namespace SharpLeNet.Core.Layers;

/// <summary>
/// Слой AvgPooling
/// </summary>
public class AvgPoolingLayer : Layer
{
    private readonly int _poolSize;
    private readonly int _stride;

    public AvgPoolingLayer(int poolSize = 2, int stride = 2)
    {
        _poolSize = poolSize;
        _stride = stride;
    }

    public override Tensor Forward(Tensor input)
    {
        if (input.Rank != 4)
            throw new ArgumentException("AvgPooling слой ожидает 4D тензор");

        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height - _poolSize) / _stride + 1;
        int outputWidth = (width - _poolSize) / _stride + 1;
        double poolArea = _poolSize * _poolSize;

        var output = new Tensor(new int[] { batchSize, channels, outputHeight, outputWidth });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        double sum = 0;

                        for (int ph = 0; ph < _poolSize; ph++)
                        {
                            for (int pw = 0; pw < _poolSize; pw++)
                            {
                                int inputH = oh * _stride + ph;
                                int inputW = ow * _stride + pw;
                                sum += input[b, c, inputH, inputW];
                            }
                        }

                        output[b, c, oh, ow] = sum / poolArea;
                    }
                }
            }
        }

        return output;
    }
}
