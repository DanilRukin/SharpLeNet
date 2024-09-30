using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Functions
{
    /// <summary>
    /// Функция активации
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IFunction<T>
    {
        /// <summary>
        /// Возвращает значение функции
        /// </summary>
        /// <param name="parameter"></param>
        /// <returns></returns>
        T Function(T parameter);

        /// <summary>
        /// Первая производная
        /// </summary>
        /// <returns></returns>
        T Derivative(T value);

        /// <summary>
        /// Вторая производная
        /// </summary>
        /// <returns></returns>
        T Derivative2(T value);
    }
}
