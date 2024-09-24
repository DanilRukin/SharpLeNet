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

            Filters = Enumerable.Repeat(new Tensor(_filtersSize, _filtersSize, _filterDepth), _filtersCount).ToList();
            _dFilters = Enumerable.Repeat(new Tensor(_filtersSize, _filtersSize, _filterDepth), filtersCount).ToList();

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
                            Filters[index][i, j, k] = random.NextDouble(); // генерируем случайное число и записываем его в элемент фильтра
                        }
                    }
                }
                Bias[index] = 0.01; // все смещения устанавливаем в 0.01
            }
        }

        public Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }

        public Tensor Backward(Tensor input)
        {
            throw new NotImplementedException();
        }
    }
}
