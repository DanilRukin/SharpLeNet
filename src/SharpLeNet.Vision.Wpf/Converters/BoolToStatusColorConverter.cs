using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class BoolToStatusColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is bool boolValue && boolValue)
        {
            return new SolidColorBrush(Color.FromRgb(74, 222, 128)); // emerald-400 for Running
        }
        return new SolidColorBrush(Color.FromRgb(156, 163, 175)); // gray-400 for Stopped/Paused
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
