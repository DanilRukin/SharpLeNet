using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class PercentageToWidthConverter : IMultiValueConverter
{
    public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
    {
        if (values.Length >= 2 && values[0] is double percentage && values[1] is double containerWidth)
        {
            // Если передан параметр, используем его как максимальное значение
            if (parameter != null && double.TryParse(parameter.ToString(), out double maxValue))
            {
                return (percentage / maxValue) * containerWidth;
            }

            // Иначе считаем, что процент уже в диапазоне 0-100
            return (percentage / 100.0) * containerWidth;
        }

        // Если значений недостаточно, возвращаем 0
        return 0.0;
    }

    public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
