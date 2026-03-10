using SharpLeNet.Vision.Wpf.ViewModels;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToColorBrushConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is LayerPaletteItemViewModel layer && parameter is string param)
        {
            var color = param switch
            {
                "bg" => layer.BackgroundGradient?.Split(' ').LastOrDefault() ?? "#000000",
                "fg" => layer.Color,
                _ => "#FFFFFF"
            };

            try
            {
                return new SolidColorBrush((Color)ColorConverter.ConvertFromString(color));
            }
            catch
            {
                return new SolidColorBrush(Colors.Transparent);
            }
        }
        return new SolidColorBrush(Colors.Transparent);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}