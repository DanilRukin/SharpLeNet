using LeNet.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Formats.Asn1.AsnWriter;

namespace LeNet.Layers
{
    public class MaxPoolingLayer
    {
        /// <summary>
        /// Размер входа
        /// </summary>
        private TensorSize _inputSize;

        /// <summary>
        /// Размер выхода
        /// </summary>
        private TensorSize _outputSize;

        /// <summary>
        /// Во сколько раз уменьшается размерность
        /// </summary>
        private int _scale;

        /// <summary>
        /// Бинарные маски входного тензора. Там, где расположен максимальный элемент, стоит 1, в других местах 0
        /// </summary>
        private Tensor _masks;

        public MaxPoolingLayer(TensorSize inputSize, int scale = 2)
        {
            _inputSize = new TensorSize()
            {
                Height = inputSize.Height,
                Width = inputSize.Width,
                Depth = inputSize.Depth,
            };

            _outputSize = new TensorSize()
            {
                Height = inputSize.Height / 2,
                Width = inputSize.Width / 2,
                Depth = inputSize.Depth
            };

            _masks = new(_inputSize);

            _scale = scale;
        }

        /// <summary>
        /// Прямое распространение сигнала. Операция макспулинга
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            Tensor output = new(_outputSize); // создаём выходной тензор

            // проходимся по каждому из каналов
            for (int d = 0; d < _inputSize.Depth; d++)
            {
                for (int i = 0; i < _inputSize.Height; i += _scale)
                {
                    for (int j = 0; j < _inputSize.Width; j += _scale)
                    {
                        int imax = i; // индекс строки максимума
                        int jmax = j; // индекс столбца максимума
                        double max = input[d, i, j]; // начальное значение максимума - значение первой клетки подматрицы

                        // проходимся по подматрице и ищем максимум и его координаты
                        for (int y = i; y < i + _scale; y++)
                        {
                            for (int x = j; x < j + _scale; x++)
                            {
                                double value = input[d, y, x]; // получаем значение входного тензора

                                // если очередное значение больше максимального
                                if (value > max)
                                {
                                    max = value; // обновляем максимум
                                    imax = i;
                                    jmax = j;
                                }    
                                    
                            }
                        }
                        output[d, i / _scale, j / _scale] = max; // записываем в выходной тензор найденный максимум
                        _masks[d, imax, jmax] = 1;
                    }
                }
            }

            return output; // возвращаем выходной тензор
        }

        /// <summary>
        /// Обратное распространение ошибки. Вычисление градиента.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dout"></param>
        /// <returns></returns>
        public Tensor Backward(Tensor input, Tensor dout)
        {
            Tensor dX = new(_inputSize); // создаём тензор для градиентов

            for (int d = 0; d < _inputSize.Depth; d++)
                for (int i = 0; i < _inputSize.Height; i++)
                    for (int j = 0; j < _inputSize.Width; j++)
                        dX[d, i, j] = dout[d, i / _scale, j / _scale] * _masks[d, i, j]; // умножаем градиенты на маску

            return dX; // возвращаем посчитанные градиенты
        }
    }
}
