using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLeNet.Core;

public static class TensorOperations
{
    /// <summary>
    /// Операция сложения тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    public static Tensor Add(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Кол-во измерений и их размерности у слагаемых " +
                "должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] + b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b, TensorOperation.Add, 
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция вычитания тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    public static Tensor Subtract(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Кол-во измерений и их размерности у слагаемых " +
                "должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] - b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b, TensorOperation.Subtract,
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция умножения тензоров (поэлементная)
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor Mul(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Кол-во измерений и их размерности у множителей " +
                "должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] * b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b,
            TensorOperation.Mul, a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция матричного умножения тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    /// <exception cref="ArgumentException"></exception>
    public static Tensor MatMul(this Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("Операция матричного умножения может быть" +
                " выполнена только для матриц!");
        if (a.Shape[1] != b.Shape[0])
            throw new ArgumentException($"Кол-во столбцов первой матрицы должно " +
                $"совпадать с кол-вом строк второй матрицы! Размерности текущих матриц: " +
                $"[{a.Shape[0]}, {a.Shape[1]}] x [{b.Shape[0]}, {b.Shape[1]}]");

        int m = a.Shape[0];
        int n = a.Shape[1];
        int p = b.Shape[1];

        double[] resultData = new double[m * p];
        int[] resultShape = new int[] { m, p };

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                resultData[i * p + j] = sum;
            }
        }

        return new Tensor(resultData, resultShape, a, b, TensorOperation.MatMul,
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Операция вычисления сигмоиды
    /// </summary>
    /// <param name="a">Тензор, для которго выполняется операция</param>
    public static Tensor Sigmoid(this Tensor a)
    {
        var resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = 1.0 / (1.0 + Math.Exp(-a.Data[i]));
        }

        return new Tensor(resultData, a.Shape, a, null, TensorOperation.Sigmoid,
            a.RequiresGrad);
    }

    /// <summary>
    /// Операция сложения всех элементов тензора. 
    /// В результате - скалярная сумма всех элементов
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется операция</param>
    public static Tensor Sum(this Tensor a)
    {
        double sum = 0;
        for (int i = 0; i < a.Size; i++)
        {
            sum += a.Data[i];
        }

        return new Tensor([sum], [1], a, null, 
            TensorOperation.Sum, a.RequiresGrad);
    }

    /// <summary>
    /// Операция отрицания
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется отрицание</param>
    public static Tensor Neg(this Tensor a)
    {
        double[] result = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            result[i] = -a.Data[i];
        }

        return new Tensor(result, a.Shape, a, null, TensorOperation.Neg,
            a.RequiresGrad);
    }

    /// <summary>
    /// Операция вычисления функции ReLU для тензора.
    /// ReLU: max(0, x)
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется вычисление ReLU</param>
    public static Tensor ReLU(this Tensor a)
    {
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = Math.Max(0, a.Data[i]);
        }

        return new Tensor(resultData, a.Shape, a, null, TensorOperation.ReLU,
            a.RequiresGrad);
    }

    /// <summary>
    /// Операция вычисления Softmax
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется вычисление Softmax</param>
    public static Tensor Softmax(this Tensor a)
    {
        if (a.Rank != 2)
            throw new NotImplementedException("Softmax поддерживается только для матриц!");

        int batchSize = a.Shape[0];
        int numClasses = a.Shape[1];

        var resultData = new double[a.Size];

        for (int i = 0; i < batchSize; i++)
        {
            // Находим максимум для численной стабильности
            double maxVal = double.MinValue;
            for (int j = 0; j < numClasses; j++)
            {
                if (a[i, j] > maxVal) maxVal = a[i, j];
            }

            // Вычисляем экспоненты
            double sumExp = 0;
            double[] exps = new double[numClasses];
            for (int j = 0; j < numClasses; j++)
            {
                exps[j] = Math.Exp(a[i, j] - maxVal);
                sumExp += exps[j];
            }

            // Нормализуем
            for (int j = 0; j < numClasses; j++)
            {
                resultData[i * numClasses + j] = exps[j] / sumExp;
            }
        }

        return new Tensor(resultData, a.Shape, a, null, TensorOperation.Softmax,
            a.RequiresGrad);
    }

    /// <summary>
    /// Вычисляет Softmax + CrossEntropy Loss (вместе для эффективности)
    /// </summary>
    /// <param name="logits"></param>
    /// <param name="labels"></param>
    /// <exception cref="ArgumentException"></exception>
    public static (Tensor softmaxOutput, Tensor loss) SoftmaxCrossEntropy(
    this Tensor logits, Tensor labels)
    {
        if (logits.Rank != 2 || labels.Rank != 2)
            throw new ArgumentException("Оба тензора должны быть матрицами!");
        if (!logits.Shape.SequenceEqual(labels.Shape))
            throw new ArgumentException("Измерения и их размерности должны совпадать!");

        int batchSize = logits.Shape[0];
        int numClasses = logits.Shape[1];

        // Вычисляем Softmax (это часть графа!)
        Tensor softmaxOutput = logits.Softmax();

        // Вычисляем Cross-Entropy Loss через операции тензоров
        // L = -mean(y * log(softmax))

        // 1. log(softmax)
        var logSoftmax = softmaxOutput.Log();

        // 2. y * log(softmax)  
        var yLogSoftmax = labels * logSoftmax;

        // 3. Суммируем по классам и батчу
        var sumPerSample = new Tensor(new int[] { batchSize });
        for (int i = 0; i < batchSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < numClasses; j++)
            {
                sum += yLogSoftmax[i, j];
            }
            sumPerSample.Data[i] = sum;
        }

        // 4. Берем отрицание и усредняем
        var negSum = -sumPerSample;
        var lossSum = negSum.Sum(); // Сумма по батчу

        // 5. Делим на размер батча
        var batchSizeTensor = new Tensor(new double[] { batchSize }, new int[] { 1 });
        var loss = lossSum / batchSizeTensor;

        return (softmaxOutput, loss);
    }

    /// <summary>
    /// Операция вычисления логарифма
    /// </summary>
    /// <param name="a">Тензор, для которого выполняется вычисление логарифма</param>
    public static Tensor Log(this Tensor a)
    {
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = Math.Log(a.Data[i]);
        }

        return new Tensor(resultData, a.Shape, a, null, TensorOperation.Log,
            a.RequiresGrad);
    }

    /// <summary>
    /// Broadcast тензора
    /// </summary>
    public static Tensor Broadcast(this Tensor a, int[] newShape)
    {
        // Простая реализация для broadcast bias в LinearLayer
        // a: [output_size]
        // newShape: [batch_size, output_size]

        double[] broadcastedData = new double[newShape.Aggregate(1, (x, y) => x * y)];
        int batchSize = newShape[0];
        int features = newShape[1];

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < features; j++)
            {
                broadcastedData[i * features + j] = a.Data[j];
            }
        }

        return new Tensor(broadcastedData, newShape, a, null, TensorOperation.Broadcast, 
            a.RequiresGrad);
    }

    /// <summary>
    /// Поэлементное деление тензоров
    /// </summary>
    /// <param name="a">Левый операнд</param>
    /// <param name="b">Правый операнд</param>
    public static Tensor Div(this Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Размерности операндов должны совпадать!");
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] / b.Data[i];
        }

        return new(resultData, a.Shape, a, b, TensorOperation.Div,
            a.RequiresGrad || b.RequiresGrad);
    }

    /// <summary>
    /// Умножение тензора на скаляр
    /// </summary>
    /// <param name="a">Тензор</param>
    /// <param name="scalar">Число для умножения элементов</param>
    public static Tensor MulScalar(this Tensor a, double scalar)
    {
        double[] resultData = new double[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] * scalar;
        }
        Tensor scalarTensor = new([scalar], [1]);

        return new(resultData, a.Shape, a, scalarTensor, TensorOperation.MulScalar,
            a.RequiresGrad);
    }

    /// <summary>
    /// Деление тензора на скаляр
    /// </summary>
    /// <param name="a">Тензор</param>
    /// <param name="scalar">число для деления</param>
    public static Tensor DivScalar(this Tensor a, double scalar) =>
        a.MulScalar(1.0 / scalar);
}
