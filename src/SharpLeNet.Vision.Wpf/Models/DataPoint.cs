namespace SharpLeNet.Vision.Wpf.Models;

public class DataPoint
{
    public DataPoint(double x, double y)
    {
        X = x;
        Y = y;
    }

    public double X { get; }
    public double Y { get; }
}
