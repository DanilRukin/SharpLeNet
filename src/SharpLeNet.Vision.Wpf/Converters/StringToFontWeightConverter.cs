using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToFontWeightConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string currentTheme && parameter is string targetTheme)
        {
            return currentTheme.Equals(targetTheme, StringComparison.OrdinalIgnoreCase)
                ? FontWeights.Bold
                : FontWeights.Normal;
        }
        return FontWeights.Normal;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
