namespace SharpLeNet.Core;

/// <summary>
/// Тензор
/// </summary>
public class Tensor : IDisposable
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
        return new Tensor((double[])Data.Clone(), (int[])Shape.Clone(), RequiresGrad);
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
        Console.WriteLine($"\n[DEBUG] Tensor.Backward() called");
        Console.WriteLine($"  Operation: {Operation}");
        Console.WriteLine($"  RequiresGrad: {RequiresGrad}");
        Console.WriteLine($"  Gradient provided: {gradient != null}");

        if (!RequiresGrad)
        {
            Console.WriteLine("  [SKIP] No grad required");
            return;
        }    
            

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
        Console.WriteLine($"[DEBUG] AddGradient called for tensor with operation: {Operation}");
        Console.WriteLine($"[DEBUG]   Current Grad is null: {Grad == null}");
        Console.WriteLine($"[DEBUG]   Gradient shape: [{string.Join(", ", gradient.Shape)}]");

        if (!Shape.SequenceEqual(gradient.Shape))
            throw new ArgumentException("Кол-во измерений и их размерность градиента " +
                "должна сопадать с кол-вом измерений и их размерностью у данного экземпляра");
        if (Grad == null)
        {
            Grad = new Tensor(gradient.Shape);
        }

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
        Console.WriteLine($"\n[DEBUG] BackwardToParents для операции {Operation}:");
        Console.WriteLine($"  LeftParent is null: {LeftParent == null}");
        Console.WriteLine($"  RightParent is null: {RightParent == null}");
        Console.WriteLine($"  Grad is null: {Grad == null}");

        switch (Operation)
        {
            case TensorOperation.Add:
                // d(L)/dA = d(L)/dC * 1
                // d(L)/dB = d(L)/dC * 1
                LeftParent?.Backward(Grad);
                RightParent?.Backward(Grad);
                break;

            case TensorOperation.Subtract:
                // d(L)/dA = d(L)/dC * 1
                // d(L)/dB = d(L)/dC * (-1)
                Console.WriteLine($"[DEBUG SUBTRACT] Grad = [{Grad?.Data[0]:F6}, {Grad?.Data[1]:F6}]");
                LeftParent?.Backward(Grad);
                if (RightParent != null)
                {
                    Tensor negativeGrad = new(Grad!.Shape);
                    negativeGrad.Fill(-1.0);
                    Tensor gradForRight = Grad! * negativeGrad;
                    RightParent.Backward(gradForRight);
                }
                break;

            case TensorOperation.Sum:
                // Для операции суммирования всех элементов в скаляр
                // ∂L/∂x_i = ∂L/∂sum * 1 (для каждого элемента)
                Console.WriteLine($"[DEBUG] Processing Sum backward, Grad[0] = {Grad?.Data[0]:F4}");
                if (LeftParent != null && Grad != null)
                {
                    // Grad - это скаляр (∂L/∂sum)
                    double scalarGrad = Grad.Data[0];

                    // Создаем тензор той же формы, что и LeftParent
                    Tensor gradForParent = new Tensor(LeftParent.Shape);

                    // Заполняем scalarGrad для всех элементов
                    gradForParent.Fill(scalarGrad);
                    
                    Console.WriteLine($"[DEBUG] Passing gradient to parent shape: [{string.Join(", ", LeftParent.Shape)}]");
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.Mul:
                // d(L)/dA = d(L)/dC * B
                // d(L)/dB = d(L)/dC * A
                Console.WriteLine($"[DEBUG MUL] Grad = [{Grad?.Data[0]:F6}, {Grad?.Data[1]:F6}]");
                Console.WriteLine($"[DEBUG MUL] LeftParent Data = [{LeftParent?.Data[0]:F6}, {LeftParent?.Data[1]:F6}]");
                Console.WriteLine($"[DEBUG MUL] RightParent Data = [{RightParent?.Data[0]:F6}, {RightParent?.Data[1]:F6}]");
                if (LeftParent != null && RightParent != null)
                {
                    Tensor gradForLeft = Grad! * RightParent;
                    Tensor gradForRight = Grad! * LeftParent;
                    LeftParent.Backward(gradForLeft);
                    RightParent.Backward(gradForRight);
                }
                break;

            case TensorOperation.Div:
                // C = A / B
                // dL/dA = dL/dC * (1/B)
                // dL/dB = dL/dC * (-A/(B^2))
                if (LeftParent != null && RightParent != null)
                {
                    // Для A: grad * (1/B)
                    var oneOverB = new Tensor(RightParent.Shape);
                    for (int i = 0; i < RightParent.Size; i++)
                    {
                        oneOverB.Data[i] = 1.0 / RightParent.Data[i];
                    }
                    var gradForLeft = Grad! * oneOverB;
                    LeftParent.Backward(gradForLeft);

                    // Для B: grad * (-A/(B^2))
                    var minusAOverBSquared = new Tensor(LeftParent.Shape);
                    for (int i = 0; i < LeftParent.Size; i++)
                    {
                        double b = RightParent.Data[i];
                        minusAOverBSquared.Data[i] = -LeftParent.Data[i] / (b * b);
                    }
                    var gradForRight = Grad! * minusAOverBSquared;
                    RightParent.Backward(gradForRight);
                }
                break;

            case TensorOperation.MulScalar:
                // C = A * scalar
                // dL/dA = dL/dC * scalar
                // scalar хранится в RightParent (тензор [1])
                if (LeftParent != null && RightParent != null && RightParent.Size == 1)
                {
                    double scalar = RightParent.Data[0];

                    // Создаем тензор со скаляром той же формы что и градиент
                    var scalarTensor = new Tensor(Grad!.Shape);
                    scalarTensor.Fill(scalar);

                    var gradForParent = Grad! * scalarTensor;
                    LeftParent.Backward(gradForParent);

                    // Для скаляра обычно градиент не вычисляем
                    // Но если нужно: dL/dscalar = sum(dL/dC * A)
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

            case TensorOperation.ReLU:
                // d(L)/dx = d(L)/dReLU * (x > 0 ? 1 : 0)
                if (LeftParent != null)
                {
                    double[] reluPrimeData = new double[Size];
                    for (int i = 0; i < Size; i++)
                    {
                        reluPrimeData[i] = LeftParent.Data[i] > 0 ? 1.0 : 0.0;
                    }
                    Tensor reluPrime = new(reluPrimeData, Shape);
                    Tensor gradForParent = Grad! * reluPrime;
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.Softmax:
                // Для softmax (не используется с CrossEntropy)
                // Производная сложная: ∂softmax_i/∂z_j = softmax_i * (δ_ij - softmax_j)
                // где δ_ij = 1 если i==j, иначе 0
                if (LeftParent != null)
                {
                    int batchSize = Shape[0];
                    int numClasses = Shape[1];
                    Tensor gradForParent = new(LeftParent.Shape);
                    for (int b = 0; b < batchSize; b++)
                    {
                        // Вычисляем градиент для каждого примера в батче
                        for (int i = 0; i < numClasses; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < numClasses; j++)
                            {
                                // ∂L/∂z_i = Σ_j (∂L/∂softmax_j * ∂softmax_j/∂z_i)
                                // ∂softmax_j/∂z_i = softmax_j * (δ_ji - softmax_i)
                                double delta_ji = (j == i) ? 1.0 : 0.0;
                                double dsoftmax_j_dz_i = this[b, j] * (delta_ji - this[b, i]);
                                sum += Grad![b, j] * dsoftmax_j_dz_i;
                            }
                            gradForParent[b, i] = sum;
                        }
                    }
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.SoftmaxCrossEntropy:
                // Магически простой градиент: dL/dz = softmax(z) - y_true
                if (LeftParent != null && RightParent != null) // LeftParent = logits, RightParent = labels
                {
                    int batchSize = LeftParent.Shape[0];
                    int numClasses = LeftParent.Shape[1];

                    // Вычисляем softmax(z) - y
                    Tensor gradForParent = new Tensor(LeftParent.Shape);

                    // Сначала вычисляем softmax
                    Tensor softmax = LeftParent.Softmax();

                    // Вычисляем градиент: softmax - labels
                    for (int i = 0; i < batchSize; i++)
                    {
                        for (int j = 0; j < numClasses; j++)
                        {
                            gradForParent[i, j] = (softmax[i, j] - RightParent[i, j]) / batchSize;
                        }
                    }
                    // Передаем градиент только к logits (labels не обучаются)
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.Log:
                // d(L)/dx = d(L)/d(log(x)) * (1/x)
                if (LeftParent != null)
                {
                    double[] logPrimeData = new double[Size];
                    for (int i = 0; i < Size; i++)
                    {
                        logPrimeData[i] = 1.0 / LeftParent.Data[i];
                    }
                    Tensor logPrime = new(logPrimeData, Shape);
                    Tensor gradForParent = Grad! * logPrime;
                    LeftParent.Backward(gradForParent);
                }
                break;

            case TensorOperation.Broadcast:
                // Когда тензор broadcast'ится (например, bias [n] -> [batch, n])
                // Градиент для оригинала = sum градиентов по broadcast dimension
                Console.WriteLine($"[DEBUG] Processing Broadcast backward");
                if (LeftParent != null && Grad != null)
                {
                    // LeftParent - оригинальный тензор (например, bias)
                    // Grad - градиент broadcasted тензора [batch, features]

                    int batchSize = Shape[0];
                    int features = Shape[1];

                    Tensor gradForParent = new Tensor(LeftParent.Shape);

                    // Суммируем градиенты по batch dimension
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int f = 0; f < features; f++)
                        {
                            gradForParent.Data[f] += Grad.Data[b * features + f];
                        }
                    }

                    Console.WriteLine($"[DEBUG] Broadcast: summing over batch dim, passing to parent");
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

    public static Tensor Random(params int[] shapes)
    {
        Random rnd = new();
        double[] randomData = new double[shapes.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < randomData.Length; i++)
        {
            randomData[i] = rnd.NextDouble();
        }
        return new Tensor(randomData, shapes);
    }

    private bool _disposed = false;

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                Data = null;
                Shape = null;
                Strides = null;
                Grad?.Dispose();
            }
            else
            {

            }
            _disposed = true;
        }
    }
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    ~Tensor()
    {
        Dispose(false);
    }

    public static Tensor operator +(Tensor a, Tensor b) => a.Add(b);

    public static Tensor operator *(Tensor a, Tensor b) => a.Mul(b);

    public static Tensor operator -(Tensor a) => a.Neg();

    public static Tensor operator -(Tensor a, Tensor b) => a.Subtract(b);

    public static Tensor operator /(Tensor a, Tensor b) => a.Div(b);

    public static Tensor operator /(Tensor a, double b) => a.DivScalar(b);

    public static Tensor operator *(Tensor a, double b) => a.MulScalar(b);

    public static Tensor operator *(double a, Tensor b) => b.MulScalar(a);
}
