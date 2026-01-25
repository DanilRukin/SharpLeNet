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
        for (int i = 0; i <  a.Size; i++)
        {
            resultData[i] = a.Data[i] + b.Data[i];
        }
        return new Tensor(resultData, a.Shape, a, b, TensorOperation.Add, 
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
}
