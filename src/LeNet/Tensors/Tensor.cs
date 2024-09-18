using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Tensors
{
    public class Tensor
    {
        public TensorSize Size { get; private set; }

        /// <summary>
        /// Произведение глубины на ширину
        /// </summary>
        private readonly int DW;

        private List<double> _values;

        public Tensor(TensorSize size)
        {
            Size = size;
            DW = size.Depth * Size.Width;
            _values = new List<double>(Size.Width * Size.Depth * Size.Height);
        }

        public Tensor(int heigth, int width, int depth):
            this(new TensorSize() { Height = heigth, Width = width, Depth = depth })
        { 
        }

        public double this[int height, int width, int depth]
        {
            get => _values[height * DW + width + Size.Depth + depth];
            set => _values[height * DW + width + Size.Depth + depth] = value;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int d = 0; d < Size.Depth; d++)
            {
                for (int i = 0; i < Size.Height; i++)
                {
                    for (int j = 0; j < Size.Width; j++)
                        builder.Append(_values[i * DW + j * Size.Depth + d] + " ");
                    builder.AppendLine();
                }
                builder.AppendLine();
            }
            return builder.ToString();
        }
    }
}
