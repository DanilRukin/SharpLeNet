using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class ConnectionPointConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is Point point)
        {
            return point;
        }
        return new Point(0, 0);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
