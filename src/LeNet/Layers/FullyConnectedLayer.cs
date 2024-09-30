using Accord.Math.Random;
using LeNet.Common;
using LeNet.Matrixs;
using LeNet.Services;
using LeNet.Tensors;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Layers
{
    public class FullyConnectedLayer
    {
        /// <summary>
        /// Входой размер
        /// </summary>
        private TensorSize _inputSize;

        /// <summary>
        /// Выходной размер
        /// </summary>
        private TensorSize _outputSize;

        /// <summary>
        /// Генератор случайных чисел с нормальным распределением
        /// </summary>
        private RandomNormal _random;

        /// <summary>
        /// Число входных нейронов
        /// </summary>
        private int _inputNeuronsCount;

        /// <summary>
        /// Число выходных нейронов
        /// </summary>
        private int _outputNeuronsCount;

        /// <summary>
        /// Тип активационной функции
        /// </summary>
        private ActivationType _activationType;

        /// <summary>
        /// Тензор производных функции активации
        /// </summary>
        private Tensor _dActivationFunction;

        /// <summary>
        /// Матрица весовых коэффициентов
        /// </summary>
        private Matrix _weights;

        /// <summary>
        /// Матрица градиентов весовых коэффициентов
        /// </summary>
        private Matrix _dWeights;

        /// <summary>
        /// Смещения
        /// </summary>
        private List<double> _bias;

        /// <summary>
        /// Градиенты смещений
        /// </summary>
        private List<double> _dBias;

        /// <summary>
        /// Инициализация весовых коэффициентов
        /// </summary>
        private void InitWeights()
        {
            for (int i = 0; i < _outputNeuronsCount; i++)
            {
                for (int j = 0; j < _inputNeuronsCount; j++)
                    _weights[i, j] = _random.NextDouble(); // генерируем очередное случайное число

                _bias[i] = 0.01; // все смещения делаем равными 0.01
            }
        }

        /// <summary>
        /// применение активационной функции
        /// </summary>
        /// <param name=""></param>
        /// <param name=""></param>
        private void Activate(Tensor output)
        {
            switch (_activationType)
            {
                case ActivationType.None:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        _dActivationFunction[0, 0, i] = 1;
                    }
                    break;
                case ActivationType.Sigmoid:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        output[0, 0, i] = 1 / (1 + Math.Exp(-output[0, 0, i]));
                        _dActivationFunction[0, 0, i] = output[0, 0, i] * (1 - output[0, 0, i]);
                    }
                    break;
                case ActivationType.Tanh:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        output[0, 0, i] = Math.Tanh(output[0, 0, i]);
                        _dActivationFunction[0, 0, i] = 1 - output[0, 0, i] * output[0, 0, i];
                    }
                    break;
                case ActivationType.ReLU:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        if (output[0, 0, i] > 0)
                        {
                            _dActivationFunction[0, 0, i] = 1;
                        }
                        else
                        {
                            output[0, 0, i] = 0;
                            _dActivationFunction[0, 0, i] = 0;
                        }
                    }
                    break;
                case ActivationType.LeakyReLU:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        if (output[0, 0, i] > 0)
                        {
                            _dActivationFunction[0, 0, i] = 1;
                        }
                        else
                        {
                            output[0, 0, i] *= 0.01;
                            _dActivationFunction[0, 0, i] = 0.01;
                        }
                    }
                    break;
                case ActivationType.ELU:
                    for (int i = 0; i < _outputNeuronsCount; i++)
                    {
                        if (output[0, 0, i] > 0)
                        {
                            _dActivationFunction[0, 0, i] = 1;
                        }
                        else
                        {
                            output[0, 0, i] = Math.Exp(output[0, 0, i]) - 1;
                            _dActivationFunction[0, 0, i] = output[0, 0, i] + 1;
                        }
                    }
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        public FullyConnectedLayer(TensorSize size, int outputs, ActivationType activationType = ActivationType.None)
        {
            _weights = new(outputs, size.Height * size.Width * size.Depth);
            _dWeights = new(outputs, size.Height * size.Width * size.Depth);
            _dActivationFunction = new(1, 1, outputs);
            _random = new RandomNormal();

            _inputSize = size;

            // вычисляем выходной размер
            _outputSize = new TensorSize()
            {
                Width = 1,
                Height = 1,
                Depth = outputs
            };
            

            _inputNeuronsCount = size.Height * size.Width * size.Depth; // запоминаем число входных нейронов
            _outputNeuronsCount = outputs; // запоминаем число выходных нейронов

            _activationType = activationType; // получаем активационную функцию

            _bias = new List<double>(outputs); // создаём вектор смещений
            _dBias = new List<double>(outputs); // создаём вектор градиентов по весам смещения

            InitWeights(); // инициализируем весовые коэффициенты
        }

        /// <summary>
        ///  Прямое распространение
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            Tensor output = new(_outputSize); // создаём выходной тензор

            // проходимся по каждому выходному нейрону
            for (int i = 0; i < _outputNeuronsCount; i++)
            {
                double sum = _bias[i]; // прибавляем смещение

                // умножаем входной тензор на матрицу
                for (int j = 0; j < _inputNeuronsCount; j++)
                    sum += _weights[i, j] * input[0, 0, j];

                output[0, 0, i] = sum;
            }

            Activate(output); // применяем активационную функцию

            return output; // возвращаем выходной тензор
        }

        /// <summary>
        /// Обратное распространение
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dout"></param>
        /// <returns></returns>
        public Tensor Backward(Tensor input, Tensor dout)
        {
            // домножаем производные на градиенты следующего слоя для сокращения количества умножений
            for (int i = 0; i < _outputNeuronsCount; i++)
                _dActivationFunction[0, 0, i] *= dout[0, 0, i];

            // вычисляем градиенты по весовым коэффициентам
            for (int i = 0; i < _outputNeuronsCount; i++)
            {
                for (int j = 0; j < _inputNeuronsCount; j++)
                    _dWeights[i, j] = _dActivationFunction[0, 0, i] * input[0, 0, j];

                _dBias[i] = _dActivationFunction[0, 0, i];
            }

            Tensor dX = new(_inputSize); // создаём тензор для градиентов по входам

            // вычисляем градиенты по входам
            for (int j = 0; j < _inputNeuronsCount; j++)
            {
                double sum = 0;

                for (int i = 0; i < _outputNeuronsCount; i++)
                    sum += _weights[i, j] * _dActivationFunction[0, 0, i];

                dX[0, 0, j] = sum; // записываем результат в тензор градиентов
            }

            return dX; // возвращаем тензор градиентов
        }

        /// <summary>
        /// Обновление весовых коэффициентов
        /// </summary>
        /// <param name="learningRate"></param>
        public void UpdateWeights(double learningRate)
        {
            for (int i = 0; i < _outputNeuronsCount; i++)
            {
                for (int j = 0; j < _inputNeuronsCount; j++)
                    _weights[i, j] -= learningRate * _dWeights[i, j];

                _bias[i] -= learningRate * _dBias[i]; // обновляем веса смещения
            }
        }
    }
}
