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

    /// <summary>
    /// Левый операнд (родительский тензор)
    /// </summary>
    public Tensor? LeftParent { get; private set; }

    /// <summary>
    /// Правый операнд (родительский тензор)
    /// </summary>
    public Tensor? RightParent { get; private set; }

    /// <summary>
    /// Операция, породившая тензор
    /// </summary>
    public TensorOperation Operation { get; private set; }

    /// <summary>
    /// Указывает, вычислен ли градиент
    /// </summary>
    public bool RequiresGrad { get; private set; }

    public Tensor(double[] data, int[] shape, Tensor? left, Tensor? right,
        TensorOperation operation, bool requiresGrad = false)
    {
        if (data.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("Общее количество элементов должно совпадать с произведением размерностей!");
        Data = data;
        Shape = (int[])shape.Clone();

        ComputeStrides();

        LeftParent = left;
        RightParent = right;
        Operation = operation;
        RequiresGrad = requiresGrad 
            || (LeftParent?.RequiresGrad == true) 
            || (RightParent?.RequiresGrad == true);

        if (RequiresGrad)
        {
            Grad = new Tensor(shape);
        }
    }

    public Tensor(double[] data, int[] shape, bool requiresGrad = false) 
        : this(data, shape, null, null, TensorOperation.None, requiresGrad)
    {
    }

    public Tensor(int[] shape, bool requiresGrad = false) 
        : this(new double[shape.Aggregate(1, (a, b) => a * b)], shape, requiresGrad)
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

    /// <summary>
    /// Обратное распространение ошибки
    /// </summary>
    /// <param name="gradient">Градиент</param>
    public void Backward(Tensor? gradient = null)
    {
        if (!RequiresGrad)
            return;

        // Если gradient == null и это скаляр (размер = 1), инициализируем как 1.0
        if (gradient == null)
        {
            if (Size != 1)
                throw new InvalidOperationException("Градиент может быть создан " +
                    "для скалярного значения");
            Grad!.Fill(1.0);
        }
        else
        {
            // Суммируем градиенты (для случая, когда тензор используется несколько раз)
            AddGradient(gradient);
        }

        BackwardToParents();
    }

    /// <summary>
    /// Выполняет добавление значений градиента к текущим значениям
    /// в <see cref="Data"/> тензора
    /// </summary>
    /// <param name="gradient">Градиент для добавления</param>
    /// <exception cref="ArgumentException"></exception>
    private void AddGradient(Tensor gradient)
    {
        if (!Shape.SequenceEqual(gradient.Shape))
            throw new ArgumentException("Кол-во измерений и их размерность градиента " +
                "должна сопадать с кол-вом измерений и их размерностью у данного экземпляра");
        for (int i = 0; i < Size; i++)
        {
            Grad!.Data[i] += gradient.Data[i];
        }
    }

    /// <summary>
    /// В зависимости от операции, вычисляет градиенты для родителей
    /// </summary>
    private void BackwardToParents()
    {
        switch (Operation)
        {
            case TensorOperation.Add:
                // d(L)/dA = d(L)/dC * 1
                // d(L)/dB = d(L)/dC * 1
                LeftParent?.Backward(Grad);
                RightParent?.Backward(Grad);
                break;
            case TensorOperation.Mul:
                // d(L)/dA = d(L)/dC * B
                // d(L)/dB = d(L)/dC * A
                if (LeftParent != null && RightParent != null)
                {
                    Tensor gradForLeft = Grad! * RightParent;
                    Tensor gradForRight = Grad! * LeftParent;
                    LeftParent.Backward(gradForLeft);
                    RightParent.Backward(gradForRight);
                }
                break;

            // Другие операции добавим позже
            case TensorOperation.MatMul:
                // dL/dA = dL/dC @ B^T
                // dL/dB = A^T @ dL/dC
                if (LeftParent != null && RightParent != null)
                {
                    Tensor gradForLeft = Grad!.MatMul(RightParent.Transpose());
                    Tensor gradForRight = LeftParent.Transpose().MatMul(Grad!);
                    LeftParent.Backward(gradForLeft);
                    RightParent.Backward(gradForRight);
                }
                break;

            case TensorOperation.Sigmoid:
                // d(L)/dx = d(L)/dσ * σ'(x)
                // где σ'(x) = σ(x) * (1 - σ(x))
                if (LeftParent != null)
                {
                    // Вычисляем σ'(x) = output * (1 - output)
                    // где output = this (результат сигмоиды)
                    double[] sigmaPrimeData = new double[Size];
                    for (int i = 0; i < Size; i++)
                    {
                        sigmaPrimeData[i] = Data[i] * (1 - Data[i]);
                    }
                    Tensor sigmaPrime = new(sigmaPrimeData, Shape);

                    // Умножаем градиент на производную
                    Tensor gradForParent = Grad! * sigmaPrime;
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.Neg:
                // d(L)/dx = d(L)/d(-x) * (-1)
                if (LeftParent != null)
                {
                    Tensor negativeGrad = new(Shape);
                    negativeGrad.Fill(-1.0);
                    Tensor gradForParent = Grad! * negativeGrad;
                    LeftParent.Backward(gradForParent);
                }
                break;

            default:
                // Листовой узел (исходные данные) - не имеет родителей
                break;
        }
    }

    /// <summary>
    /// Обнуляет градиент
    /// </summary>
    public void ZeroGrad() => Grad?.Fill(0.0);

    public static Tensor operator +(Tensor a, Tensor b) => a.Add(b);

    public static Tensor operator *(Tensor a, Tensor b) => a.Mul(b);

    public static Tensor operator -(Tensor a) => a.Neg();
}
