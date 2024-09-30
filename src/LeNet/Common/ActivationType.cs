using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet.Common
{
    public enum ActivationType
    {
        ///<summary> без активации </summary>
        None,

        ///<summary> сигмоидальная функция </summary>
        Sigmoid,

        ///<summary> гиперболический тангенс </summary>
        Tanh,

        ///<summary> выпрямитель </summary>
        ReLU,

        ///<summary> выпрямитель с утечкой </summary>
        LeakyReLU,

        ///<summary> экспоненциальный выпрямитель </summary>
        ELU
    }
}
