using SharpLeNet.Vision.Wpf.Models;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class LogLevelToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is LogLevel level)
        {
            return level switch
            {
                LogLevel.Debug => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#60a5fa")),   // Синий
                LogLevel.Info => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#d4af37")),    // Золотой
                LogLevel.Warning => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#fdba74")), // Оранжевый
                LogLevel.Error => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#fda4af")),   // Красный
                _ => new SolidColorBrush(Colors.White)
            };
        }
        return new SolidColorBrush(Colors.White);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
