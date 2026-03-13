using SharpLeNet.Vision.Wpf.Models;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.Converters;

public class DataPointsToPointCollectionConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var points = new PointCollection();

        if (value is IEnumerable<DataPoint> dataPoints && dataPoints.Any())
        {
            // Нормализуем значения для отображения на графике
            double minX = dataPoints.Min(p => p.X);
            double maxX = dataPoints.Max(p => p.X);
            double minY = dataPoints.Min(p => p.Y);
            double maxY = dataPoints.Max(p => p.Y);

            // Добавляем небольшой отступ
            double epochRange = maxX - minX;
            double valueRange = maxY - minY;

            if (epochRange == 0) epochRange = 1;
            if (valueRange == 0) valueRange = 1;

            foreach (var point in dataPoints)
            {
                // Нормализуем координаты в диапазон [0, 100] для удобства отображения
                double x = ((point.X - minX) / epochRange) * 100;
                double y = 100 - ((point.Y - minY) / valueRange) * 100; // Инвертируем Y

                points.Add(new Point(x, y));
            }
        }

        return points;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
