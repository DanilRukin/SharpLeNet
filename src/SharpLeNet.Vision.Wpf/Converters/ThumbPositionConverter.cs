using System.Globalization;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

/// <summary>
/// Конвертирует значение слайдера в позицию Thumb
/// </summary>
public class ThumbPositionConverter : IMultiValueConverter
{
    public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
    {
        if (values.Length >= 3 &&
            values[0] is double value &&
            values[1] is double minimum &&
            values[2] is double maximum)
        {
            var range = maximum - minimum;
            if (range == 0) return 0.0;

            var percent = (value - minimum) / range;
            return percent;
        }
        return 0.0;
    }

    public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
