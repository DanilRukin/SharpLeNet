using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Matrixs
{
    public class Matrix
    {
        private int _rows;
        private int _cols;
        private List<List<double>> _values;

        public Matrix(int rows, int cols)
        {
            _rows = rows;
            _cols = cols;
            _values = Enumerable.Repeat(new List<double>(cols), rows).ToList();
        }

        public double this[int i, int j]
        {
            get => _values[i][j];
            set => _values[i][j] = value;
        }
    }
}
