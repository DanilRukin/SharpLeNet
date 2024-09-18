using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet
{
    public class MathCore
    {
        public static double[,] MaxPooling(double[,] matrix, int poolSize)
        {
            int height = matrix.GetLength(0);
            int width = matrix.GetLength(1);
            if (matrix.GetLength(0) % poolSize != 0)
                throw new ArgumentException($"Невозможно провести операцию макспулинга ядром " +
                    $"{poolSize}*{poolSize} для матрицы {height}*{width}");
            int resultMatrixDimensionSize = height / poolSize;
            double[,] result = new double[resultMatrixDimensionSize, resultMatrixDimensionSize];
            double maxValue = double.MinValue;
            for (int i = 0; i < resultMatrixDimensionSize; i++)
            {
                for (int ii = 0; ii < resultMatrixDimensionSize; ii++)
                {
                    maxValue = 0;
                    // обход по ядру
                    for (int y = i * poolSize; y < i * poolSize + poolSize; y++) // по строкам
                    {
                        for (int x = ii * poolSize; x < ii * poolSize + poolSize; x++) // по столбцам
                        {
                            if (matrix[y, x] > maxValue)
                            {
                                maxValue = matrix[y, x];
                            }
                        }
                    }
                    result[i, ii] = maxValue;
                }
            }
            return result;
        }

        public static double[,] Convolution(double[,] matrix, double[,] kernel, int stride, int padding, ExpandMethod expandMethod)
        {
            int kernelHeight = kernel.GetLength(0);
            int kernelWidth = kernel.GetLength(1);
            int sourceHeight = matrix.GetLength(0);
            int sourceWidth = matrix.GetLength(1);
            int resultHeight = (sourceHeight - kernelHeight + 2 * padding) / stride + 1;
            int resultWidth = (sourceWidth - kernelWidth + 2 * padding) / stride + 1;
            double[,] expanded = Expand(matrix, padding, expandMethod);
            double[,] result = new double[resultHeight, resultWidth];
            int y, x;
            for (int i = 0; i < resultHeight; i++)
            {
                y = i * stride + kernelHeight / 2;
                for (int j = 0; j < resultWidth; j++)
                {
                    x = j * stride + kernelWidth / 2;
                    result[i, j] = GetConvolutionResult(expanded, y, x, kernel);
                }
            }
            return result;
        }

        private static double GetConvolutionResult(double[,] source, int kernelTargetRow, int kernelTargetCol, double[,] kernel)
        {
            // Условия:
            // всегда указывается валидный центр для установки ядра, чтобы не было выхода за пределы source
            int height = kernel.GetLength(0);
            int width = kernel.GetLength(1);
            if (height != width)
                throw new ArgumentException("Ядро должно быть квадратным");
            if (height % 2 == 0)
                throw new ArgumentException("У ядра должен быть нечетный размер, например, 3*3");
            double result = 0;
            int kernelPadding = width / 2; // сколько надо отступить от центра ядра, чтобы попасть в его край
            // проход ядром
            // вычисляем координаты левого верхнего угла source матрицы относительно центра постановки ядра
            int y = kernelTargetRow - kernelPadding;
            int x = kernelTargetCol - kernelPadding;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result += source[y + i, x + j] * kernel[i, j];
                }
            }
            return result;
        }

        public static double[,] Expand(double[,] matrix, int count, ExpandMethod expandMethod)
        {
            int heigth = matrix.GetLength(0);
            int width = matrix.GetLength(1);
            if (heigth < count || width < count)
                throw new InvalidOperationException($"Невозможно расширить матрицу {heigth}*{width} на" +
                    $" {count} элементов в каждую сторону");
            /*
                ****
                ****
              **++++**
              **++++**   // сначала делаем вот такую матрицу
              **++++**
                ****
                ****
             */
            int resultHeight = heigth + count * 2;
            int resultWidth = width + count * 2;
            double[,] result = new double[resultHeight, resultWidth];
            int x, y;
            // копируем старую матрицу в центр новой
            for (int i = 0; i < heigth; i++)
            {
                y = count + i; // индекс строки в новой матрице
                for (int j = 0; j < width; j++)
                {
                    x = count + j; // индекс столбца в новой матрице
                    result[y, x] = matrix[i, j];
                }
            }
            // если ExpandMethod = WithZeros, то ничего не делаем
            if (expandMethod == ExpandMethod.WithZeros)
            {
                return result;
            }
            if (expandMethod == ExpandMethod.Mirror)
            {
                
                for (int i = 0; i < count; i++)
                {
                    // верх
                    y = count + i;
                    for (int j = 0; j < width; j++)
                    {
                        x = count + j;
                        result[count - i - 1, x] = result[y, x];
                    }
                    // низ
                    y = heigth + count + i;
                    for (int j = 0; j < width; j++)
                    {
                        x = count + j;
                        result[y, x] = result[heigth + count - i - 1, x];
                    }
                }

                for (int i = 0; i < resultHeight; i++)
                {
                    // лево
                    for (int j = 0; j < count; j++)
                    {
                        result[i, count - j - 1] = result[i, count + j];
                    }
                    // право
                    for (int j = 0; j < count; j++)
                    {
                        result[i, resultWidth - j - 1] = result[i, width + j];
                    }
                }
            }
            else // Copy
            {
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        // верх
                        result[i, j + count] = result[i + count, j + count];

                        // низ
                        result[heigth + count + i, j + count] = result[heigth + i, j + count];
                    }
                }
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < resultHeight; j++)
                    {
                        // лево
                        result[j, i] = result[j, count + i];

                        // право
                        result[j, count + width + i] = result[j, width + i];
                    }
                }
            }
            return result;
        }

        public enum ExpandMethod
        {
            WithZeros, Mirror, Copy
        }

        public static double[] Flatten(double[,] source)
        {
            if (source == null)
                return new double[0];

            int height = source.GetLength(0);
            int width = source.GetLength(1);
            if (height == 0 && width == 0)
                return new double[0];

            int size = source.Length;
            double[] result = new double[size];

            int write = 0;
            for (int i = 0; i <= source.GetUpperBound(0); i++)
            {
                for (int z = 0; z <= source.GetUpperBound(1); z++)
                {
                    result[write++] = source[i, z];
                }
            }
            return result;
        }

        public static double[] SoftMax(double[] source) => Accord.Math.Special.Softmax(source);
    }
}
