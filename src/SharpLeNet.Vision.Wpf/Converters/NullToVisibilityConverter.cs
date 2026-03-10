using System.Globalization;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class NullToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool isVisible = value != null;

        if (parameter?.ToString() == "inverse")
            isVisible = !isVisible;

        return isVisible ? System.Windows.Visibility.Visible : System.Windows.Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
