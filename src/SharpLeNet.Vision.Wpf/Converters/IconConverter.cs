using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace SharpLeNet.Vision.Wpf.Converters;

public class IconConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        string iconName = value as string ?? string.Empty;

        // Return appropriate Path geometry based on icon name
        return iconName switch
        {
            "ConvIcon" => Application.Current.FindResource("ConvIcon"),
            "PoolIcon" => Application.Current.FindResource("PoolIcon"),
            "LinearIcon" => Application.Current.FindResource("LinearIcon"),
            "ActivationIcon" => Application.Current.FindResource("ActivationIcon"),
            "FlattenIcon" => Application.Current.FindResource("FlattenIcon"),
            "ImageIcon" => Application.Current.FindResource("ImageIcon"),
            _ => null
        };
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
