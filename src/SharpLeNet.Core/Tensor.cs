namespace SharpLeNet.Core;

/// <summary>
/// Тензор
/// </summary>
public class Tensor
{
    /// <summary>
    /// Основные данные
    /// </summary>
    public double[] Data { get; private set; }

    /// <summary>
    /// Форма тензора (размерности по каждому измерению)
    /// </summary>
    public int[] Shape { get; private set; }

    /// <summary>
    /// Набор смещений по каждому измерению
    /// </summary>
    /// <remarks>
    /// Strides[i] показывает, насколько элементов надо сместиться в
    /// <see cref="Data"/> при увеличении i-го индекса на 1
    /// </remarks>
    public int[] Strides { get; private set; }

    /// <summary>
    /// Общее кол-во элементов. Произведение всех размерностей.
    /// </summary>
    public int Size => Data.Length;

    /// <summary>
    /// Ранг - количество измерений.
    /// </summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Градиент. Тензор той же формы.
    /// </summary>
    public Tensor? Grad { get; set; }

    public Tensor(double[] data, int[] shape)
    {
        if (data.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("Общее количество элементов должно совпадать с произведением размерностей!");
        Data = data;
        Shape = (int[])shape.Clone();

        ComputeStrides();
    }

    public Tensor(int[] shape) : this(new double[shape.Aggregate(1, (a, b) => a * b)], shape)
    {
    }

    /// <summary>
    /// Вычисление страйдов
    /// </summary>
    private void ComputeStrides()
    {
        Strides = new int[Rank];
        if (Rank == 0)
            return;

        Strides[Rank - 1] = 1;
        for (int i = Rank - 2; i >= 0; i--)
        {
            Strides[i] = Strides[i + 1] * Shape[i + 1];
        }
    }

    public double this[params int[] indices]
    {
        get
        {
            int index = CalculateOneDimensionIndexFromManyDimensionIndices(indices);
            return Data[index];
        }
        set
        {
            int index = CalculateOneDimensionIndexFromManyDimensionIndices(indices);
            Data[index] = value;
        }
    }

    /// <summary>
    /// Вычисляет итоговый индекс в одномерном массиве <see cref="Data"/>, исходя
    /// из переданных многомерных индексов
    /// </summary>
    /// <param name="indices">Индексы в каждом измерении</param>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="IndexOutOfRangeException"></exception>
    private int CalculateOneDimensionIndexFromManyDimensionIndices(params int[] indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException($"Количество индексов должно совпадать к рангом тензора." +
                $" Индексов получено = {indices.Length}. Ожидалось = {Rank}");
        int index = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] >= Shape[i])
                throw new IndexOutOfRangeException($"Индекс {indices[i]} превышает " +
                    $"размерность измерения {i}. Размерность = {Shape[i]}");
            index += indices[i] * Strides[i];
        }
        return index;
    }

    /// <summary>
    /// Изменяет форму тензора
    /// </summary>
    /// <param name="newShape">Новые размерности</param>
    public Tensor Reshape(params int[] newShape)
    {
        int newTensorSize = newShape.Aggregate(1, (a, b) => a * b);
        if (Size != newTensorSize)
            throw new ArgumentException($"Новая размерность тензора не соответствует" +
                $" предыдущей размерности!. Новая размерность = {newTensorSize}. " +
                $"Предыдущая размерность = {Size}");
        return new Tensor(Data, newShape);
    }

    /// <summary>
    /// Транспонирование тензора (для матрицы)
    /// </summary>
    public Tensor Transpose()
    {
        if (Rank != 2)
            throw new InvalidOperationException("Транспонирование поддерживается только для 2D тензоров (матриц)");
        Tensor transposed = new Tensor([Shape[1], Shape[0]]);
        for (int i = 0; i < Shape[0]; i++)
        {
            for (int j = 0; j < Shape[1]; j++)
            {
                transposed[j, i] = this[i, j];
            }
        }
        return transposed;
    }

    /// <summary>
    /// Клонирует тензор (глубокое копирование)
    /// </summary>
    public Tensor Clone()
    {
        return new Tensor((double[])Data.Clone(), (int[])Shape.Clone());
    }

    /// <summary>
    /// Заполняет тензор значениями
    /// </summary>
    /// <param name="value">Значение, которым будет заполнен весь тензор</param>
    public void Fill(double value)
    {
        for (int i = 0; i < Size; i++)
        {
            Data[i] = value;
        }
    }

    /// <summary>
    /// Создает тензор, инициализированный нулями
    /// </summary>
    /// <param name="shape">Размерности тензора</param>
    public static Tensor Zeros(params int[] shape) => new Tensor(shape);

    /// <summary>
    /// Создает тензор, инициализированный единицами
    /// </summary>
    /// <param name="shape">Размерности тензора</param>
    public static Tensor Ones(params int[] shape)
    {
        Tensor result = new Tensor(shape);
        result.Fill(1.0);
        return result;
    }
}
