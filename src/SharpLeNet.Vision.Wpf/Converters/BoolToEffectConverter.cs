using System.Globalization;
using System.Windows.Data;
using System.Windows.Media.Effects;

namespace SharpLeNet.Vision.Wpf.Converters;

public class BoolToEffectConverter : IValueConverter
{
    private static readonly DropShadowEffect _glowEffect = new DropShadowEffect
    {
        BlurRadius = 15,
        Opacity = 0.3,
        ShadowDepth = 0,
        Color = System.Windows.Media.Color.FromRgb(212, 175, 55) // Gold500
    };

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is bool boolValue && boolValue)
        {
            return _glowEffect;
        }

        // Возвращаем null для отсутствия эффекта
        return null;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
