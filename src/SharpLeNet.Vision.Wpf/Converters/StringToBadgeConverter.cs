using SharpLeNet.Vision.Wpf.ViewModels;
using System.Globalization;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToBadgeConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is LayerPaletteItemViewModel layer)
        {
            return layer.Badge;
        }
        return string.Empty;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}