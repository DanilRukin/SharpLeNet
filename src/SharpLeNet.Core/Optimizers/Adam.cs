namespace SharpLeNet.Core.Optimizers;

/// <summary>
/// Adam оптимизатор
/// </summary>
public class Adam : Optimizer
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private int _step;

    private readonly List<Tensor> _m;
    private readonly List<Tensor> _v;

    public Adam(List<Tensor> parameters, double learningRate = 0.001, double beta1 = 0.9,
        double beta2 = 0.999, double epsilon = 1e-8) : base(parameters, learningRate)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _step = 0;

        _m = new List<Tensor>();
        _v = new List<Tensor>();

        foreach (var param in parameters)
        {
            _m.Add(new Tensor(param.Shape));
            _v.Add(new Tensor(param.Shape));
        }
    }

    public override void Step()
    {
        _step++;
        for (int i = 0; i < _parameters.Count; i++)
        {
            Tensor param = _parameters[i];
            Tensor? grad = param.Grad;

            if (grad == null) continue;

            Tensor m = _m[i];
            Tensor v = _v[i];

            // Обновить смещенную оценку первого момента
            for (int j = 0; j < param.Size; j++)
            {
                m.Data[j] = _beta1 * m.Data[j] + (1 - _beta1) * grad.Data[j];
            }

            // Обновить смещенную оценку второго момента
            for (int j = 0; j < param.Size; j++)
            {
                v.Data[j] = _beta2 * v.Data[j] + (1 - _beta2) * grad.Data[j] * grad.Data[j];
            }

            // Вычисляем скорректированную по смещению оценку первого момента.
            double mHat = 1.0 / (1.0 - Math.Pow(_beta1, _step));
            double vHat = 1.0 / (1.0 - Math.Pow(_beta2, _step));

            // Обновляем параметры
            for (int j = 0; j < param.Size; j++)
            {
                double mCorrected = m.Data[j] * mHat;
                double vCorrected = v.Data[j] * vHat;

                param.Data[j] -= _learningRate * mCorrected / (Math.Sqrt(vCorrected) + _epsilon);
            }
        }
    }

    public override void ZeroGrad()
    {
        foreach (var param in _parameters)
        {
            param.ZeroGrad();
        }
    }
}
