using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class BoolToGoldConverter : IValueConverter
{
    private static readonly SolidColorBrush _goldBrush = new(Color.FromRgb(255, 215, 0)); // Gold400
    private static readonly SolidColorBrush _whiteBrush = new(Color.FromRgb(255, 255, 255)); // White

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is bool boolValue && boolValue)
        {
            return _goldBrush;
        }
        return _whiteBrush;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
