using System.Globalization;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class BoolToOpacityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is bool boolValue && boolValue)
        {
            return 0.6; // Dimmed
        }
        return 1.0; // Full opacity
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
