using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string currentTheme && parameter is string targetTheme)
        {
            return currentTheme.Equals(targetTheme, StringComparison.OrdinalIgnoreCase)
                ? Visibility.Visible
                : Visibility.Collapsed;
        }
        return Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
