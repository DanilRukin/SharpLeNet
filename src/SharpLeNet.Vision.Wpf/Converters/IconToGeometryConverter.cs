using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class IconToGeometryConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string iconName)
        {
            var geometry = Application.Current.TryFindResource(iconName + "IconGeometry") as Geometry;
            return geometry ?? Geometry.Parse("M0,0");
        }
        return Geometry.Parse("M0,0");
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
