using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string colorName)
        {
            return colorName switch
            {
                "Gold" => Color.FromRgb(255, 215, 0),
                "Emerald" => Color.FromRgb(16, 185, 129),
                "Cyan" => Color.FromRgb(6, 182, 212),
                "Rose" => Color.FromRgb(244, 114, 182),
                "Gray" => Color.FromRgb(156, 163, 175),
                _ => Colors.Transparent
            };
        }
        return Colors.Transparent;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
