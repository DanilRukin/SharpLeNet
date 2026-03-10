using System.Globalization;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class FlopsToStringConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double flops)
        {
            if (flops < 1e3)
                return $"{flops:F0}";
            if (flops < 1e6)
                return $"{flops / 1e3:F1}K";
            if (flops < 1e9)
                return $"{flops / 1e6:F1}M";
            return $"{flops / 1e9:F1}G";
        }
        return "0";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
