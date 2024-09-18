using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Tensors
{
    public struct TensorSize
    {
        private int _height;
        public int Height
        {
            get => _height;
            set
            {
                ThrowIfInvalid(value);
                _height = value;
            }
        }

        private int _width;
        public int Width
        {
            get => _width;
            set
            {
                ThrowIfInvalid(value);
                _width = value;
            }
        }

        private int _depth;
        public int Depth
        {
            get => _depth;
            set
            {
                ThrowIfInvalid(value);
                _depth = value;
            }
        }

        private static void ThrowIfInvalid(int value, [CallerMemberName] string propertyName = "")
        {
            if (value < 0)
                throw new ArgumentException($"Значение не может быть меньше 0 ({propertyName} = {value})");
        }
    }
}
