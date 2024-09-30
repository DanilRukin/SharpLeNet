using Accord.Math.Random;
using LeNet.Services;
using LeNet.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Layers
{
    public class ConvolutionLayer
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
        /// Фильтры (W)
        /// </summary>
        private List<Tensor> Filters { get; set; }

        /// <summary>
        /// Смещения (b)
        /// </summary>
        private List<double> Bias { get; set; }

        /// <summary>
        /// Градиент фильтров (dW)
        /// </summary>
        private List<Tensor> _dFilters { get; set; }

        /// <summary>
        /// Градиент смещений (db)
        /// </summary>
        private List<double> _dBias { get; set; }

        /// <summary>
        /// Паддинг P
        /// </summary>
        private int _padding;

        /// <summary>
        /// Шаг свертки S
        /// </summary>
        private int _stride;

        /// <summary>
        /// Количество фильтров fc
        /// </summary>
        private int _filtersCount;

        /// <summary>
        /// Размер фильтров fs
        /// </summary>
        private int _filtersSize;

        /// <summary>
        /// Глубина фильтров fd
        /// </summary>
        private int _filterDepth;

        public ConvolutionLayer(TensorSize size, int filtersCount, int filtersSize, int padding, int stride)
        {
            _inputSize = size;

            _outputSize = new TensorSize()
            {
                Width = (size.Width - filtersSize + 2 * padding) / stride + 1,
                Depth = filtersCount,
                Height = (size.Height - filtersSize + 2 * padding) / stride + 1
            };

            _padding = padding;
            _stride = stride;

            _filtersCount = filtersCount;
            _filtersSize = filtersSize;
            _filterDepth = size.Depth;

            Filters = Enumerable.Repeat(new Tensor(_filterDepth, _filtersSize, _filtersSize), _filtersCount).ToList();
            _dFilters = Enumerable.Repeat(new Tensor(_filterDepth, _filtersSize, _filtersSize), filtersCount).ToList();

            Bias = Enumerable.Repeat(0d, _filtersCount).ToList();
            _dBias = Enumerable.Repeat(0d, _filtersCount).ToList();

            InitializeWeights();
        }


        private void InitializeWeights()
        {
            RandomNormal random = new RandomNormal();
            // проходимся по каждому из фильтров
            for (int index = 0; index < _filtersCount; index++)
            {
                for (int i = 0; i < _filtersSize; i++)
                {
                    for (int j = 0; j < _filtersSize; j++)
                    {
                        for (int k = 0; k < _filterDepth; k++)
                        {
                            Filters[index][k, i, j] = random.NextDouble(); // генерируем случайное число и записываем его в элемент фильтра
                        }
                    }
                }
                Bias[index] = 0.01; // все смещения устанавливаем в 0.01
            }
        }

        /// <summary>
        /// Прямое распространение сигнала, сама операция свертки
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            Tensor output = new Tensor(_outputSize);
            // проходимся по каждому из фильтров
            for (int f = 0; f < _filtersCount; f++)
            {
                for (int y = 0; y < _outputSize.Height; y++)
                {
                    for (int x = 0; x < _outputSize.Width; x++)
                    {
                        double sum = Bias[f]; // сразу прибавляем смещение

                        // проходимся по фильтрам
                        for (int i = 0; i < _filtersSize; i++)
                        {
                            for (int j = 0; j < _filtersSize; j++)
                            {
                                int i0 = _stride * y + i - _padding;
                                int j0 = _stride * x + j - _padding;

                                // поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
                                if (i0 < 0 || i0 >= _inputSize.Height || j0 < 0 || j0 >= _inputSize.Width)
                                    continue;

                                // проходимся по всей глубине тензора и считаем сумму
                                for (int c = 0; c < _filterDepth; c++)
                                    sum += input[c, i0, j0] * Filters[f][c, i, j];
                            }
                        }

                        output[f, y, x] = sum; // записываем результат свёртки в выходной тензор
                    }
                }
            }

            return output; // возвращаем выходной тензор
        }

        /// <summary>
        /// Обратное распространение ошибки, вычисление градиентов
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dout"></param>
        /// <returns></returns>
        public Tensor Backward(Tensor input, Tensor dout)
        {
            // размер дельт
            TensorSize size = new TensorSize
            {
                // расчитываем размер для дельт
                Height = _stride * (_outputSize.Height - 1) + 1,
                Width = _stride * (_outputSize.Width - 1) + 1,
                Depth = _outputSize.Depth,
            };            

            Tensor deltas = new(size); // создаём тензор для дельт

            // расчитываем значения дельт
            for (int d = 0; d < size.Depth; d++)
                for (int i = 0; i < _outputSize.Height; i++)
                    for (int j = 0; j < _outputSize.Width; j++)
                        deltas[d, i * _stride, j * _stride] = dout[d, i, j];

            // расчитываем градиенты весов фильтров и смещений
            for (int f = 0; f < _filtersCount; f++)
            {
                for (int y = 0; y < size.Height; y++)
                {
                    for (int x = 0; x < size.Width; x++)
                    {
                        double delta = deltas[f, y, x]; // запоминаем значение градиента

                        for (int i = 0; i < _filtersSize; i++)
                        {
                            for (int j = 0; j < _filtersSize; j++)
                            {
                                int i0 = i + y - _padding;
                                int j0 = j + x - _padding;

                                // игнорируем выходящие за границы элементы
                                if (i0 < 0 || i0 >= _inputSize.Height || j0 < 0 || j0 >= _inputSize.Width)
                                    continue;

                                // наращиваем градиент фильтра
                                for (int c = 0; c < _filterDepth; c++)
                                    _dFilters[f][c, i, j] += delta * input[c, i0, j0];
                            }
                        }

                        _dBias[f] += delta; // наращиваем градиент смещения
                    }
                }
            }

            int pad = _filtersSize - 1 - _padding; // заменяем величину дополнения
            Tensor dX = new(_inputSize); // создаём тензор градиентов по входу

            // расчитываем значения градиента
            for (int y = 0; y < _inputSize.Height; y++)
            {
                for (int x = 0; x < _inputSize.Width; x++)
                {
                    for (int c = 0; c < _filterDepth; c++)
                    {
                        double sum = 0; // сумма для градиента

                        // идём по всем весовым коэффициентам фильтров
                        for (int i = 0; i < _filtersSize; i++)
                        {
                            for (int j = 0; j < _filtersSize; j++)
                            {
                                int i0 = y + i - pad;
                                int j0 = x + j - pad;

                                // игнорируем выходящие за границы элементы
                                if (i0 < 0 || i0 >= size.Height || j0 < 0 || j0 >= size.Width)
                                    continue;

                                // суммируем по всем фильтрам
                                for (int f = 0; f < _filtersCount; f++)
                                    sum += Filters[f][c, _filtersSize - 1 - i, _filtersSize - 1 - j] * deltas[f, i0, j0]; // добавляем произведение повёрнутых фильтров на дельты
                            }
                        }

                        dX[c, y, x] = sum; // записываем результат в тензор градиента
                    }
                }
            }

            return dX; // возвращаем тензор градиентов
        }

        /// <summary>
        /// Обновление весов и обнуление градиентов
        /// </summary>
        /// <param name="learningRate"></param>
        public void UpdateWeights(double learningRate)
        {
            for (int index = 0; index < _filtersCount; index++)
            {
                for (int i = 0; i < _filtersSize; i++)
                {
                    for (int j = 0; j < _filtersSize; j++)
                    {
                        for (int d = 0; d < _filterDepth; d++)
                        {
                            Filters[index][d, i, j] -= learningRate * _dFilters[index][d, i, j]; // вычитаем градиент, умноженный на скорость обучения
                            _dFilters[index][d, i, j] = 0; // обнуляем градиент фильтра
                        }
                    }
                }

                Bias[index] -= learningRate * _dBias[index]; // вычитаем градиент, умноженный на скорость обучения
                _dBias[index] = 0; // обнуляем градиент веса смещения
            }
        }
    }
}
