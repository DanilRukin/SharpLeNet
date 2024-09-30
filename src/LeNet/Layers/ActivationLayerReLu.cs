using LeNet.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Layers
{
    public class ActivationLayerReLu
    {
        private TensorSize _size;

        public ActivationLayerReLu(TensorSize size)
        {
            _size = size;
        }

        public Tensor Forward(Tensor input)
        {
            Tensor output = new(_size); // создаём выходной тензор

            // проходимся по всем значениям входного тензора
            for (int i = 0; i < _size.Height; i++)
                for (int j = 0; j < _size.Width; j++)
                    for (int k = 0; k < _size.Depth; k++)
                        output[k, i, j] = input[k, i, j] > 0 ? input[k, i, j] : 0; // вычисляем значение функции активации

            return output; // возвращаем выходной тензор
        }

        public Tensor Backward(Tensor input, Tensor dout)
        {
            Tensor dX = new(_size); // создаём тензор градиентов

            // проходимся по всем значениям тензора градиентов
            for (int i = 0; i < _size.Height; i++)
                for (int j = 0; j < _size.Width; j++)
                    for (int k = 0; k < _size.Depth; k++)
                        dX[k, i, j] = dout[k, i, j] * (input[k, i, j] > 0 ? 1 : 0); // умножаем градиенты следующего слоя на производную функции активации

            return dX; // возвращаем тензор градиентов
        }
    }
}
