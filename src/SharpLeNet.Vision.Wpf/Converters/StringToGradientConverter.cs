using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class StringToGradientConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string gradientString && gradientString.StartsWith("LinearGradient"))
        {
            try
            {
                var parts = gradientString.Split(' ');
                if (parts.Length >= 6)
                {
                    var gradient = new LinearGradientBrush
                    {
                        StartPoint = new Point(double.Parse(parts[1]), double.Parse(parts[2])),
                        EndPoint = new Point(double.Parse(parts[3]), double.Parse(parts[4]))
                    };

                    for (int i = 5; i < parts.Length; i += 2)
                    {
                        if (i + 1 < parts.Length)
                        {
                            var color = (Color)ColorConverter.ConvertFromString(parts[i]);
                            var offset = double.Parse(parts[i + 1]);
                            gradient.GradientStops.Add(new GradientStop(color, offset));
                        }
                    }
                    return gradient;
                }
            }
            catch
            {
                // Fall through to default
            }
        }
        return new LinearGradientBrush(Colors.Gray, Colors.DarkGray, 90);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
