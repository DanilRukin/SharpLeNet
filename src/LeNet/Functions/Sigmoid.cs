using Accord.Neuro;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Functions
{
    public class Sigmoid : IFunction<double>
    {
        private SigmoidFunction _function;
        private SigmoidFunction SigmoidFunction => _function ??= new SigmoidFunction(Alpha);


        private double _alpha = 0;
        public double Alpha
        {
            get => _alpha;
            set
            {
                _alpha = value;
                SigmoidFunction.Alpha = _alpha;
            }
        }

        public Sigmoid(double alpha)
        {
            _alpha = alpha;
        }


        public double Derivative(double value) => SigmoidFunction.Derivative(value);

        public double Derivative2(double value) => SigmoidFunction.Derivative2(value);

        public double Function(double parameter) => SigmoidFunction.Function(parameter);
    }
}
