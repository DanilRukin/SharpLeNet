using System.Windows;
using System.Windows.Media;
using System.Windows.Shapes;

namespace SharpLeNet.Vision.Wpf.Views.Controls;

public class ConnectionLine : Shape
{
    static ConnectionLine()
    {
        StrokeProperty.OverrideMetadata(typeof(ConnectionLine), new FrameworkPropertyMetadata(Brushes.Transparent));
        FillProperty.OverrideMetadata(typeof(ConnectionLine), new FrameworkPropertyMetadata(Brushes.Transparent));
    }

    public static readonly DependencyProperty StartProperty =
        DependencyProperty.Register("Start", typeof(Point), typeof(ConnectionLine),
            new FrameworkPropertyMetadata(default(Point), FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty EndProperty =
        DependencyProperty.Register("End", typeof(Point), typeof(ConnectionLine),
            new FrameworkPropertyMetadata(default(Point), FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty Control1Property =
        DependencyProperty.Register("Control1", typeof(Point), typeof(ConnectionLine),
            new FrameworkPropertyMetadata(default(Point), FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty Control2Property =
        DependencyProperty.Register("Control2", typeof(Point), typeof(ConnectionLine),
            new FrameworkPropertyMetadata(default(Point), FrameworkPropertyMetadataOptions.AffectsRender));

    public Point Start
    {
        get => (Point)GetValue(StartProperty);
        set => SetValue(StartProperty, value);
    }

    public Point End
    {
        get => (Point)GetValue(EndProperty);
        set => SetValue(EndProperty, value);
    }

    public Point Control1
    {
        get => (Point)GetValue(Control1Property);
        set => SetValue(Control1Property, value);
    }

    public Point Control2
    {
        get => (Point)GetValue(Control2Property);
        set => SetValue(Control2Property, value);
    }

    protected override Geometry DefiningGeometry
    {
        get
        {
            var geometry = new PathGeometry();
            var figure = new PathFigure { StartPoint = Start };

            // Create cubic bezier curve
            var segment = new BezierSegment(Control1, Control2, End, true);
            figure.Segments.Add(segment);
            geometry.Figures.Add(figure);

            return geometry;
        }
    }

    // Calculate point on bezier curve at t [0,1]
    public Point GetPointAt(double t)
    {
        double u = 1 - t;
        double tt = t * t;
        double uu = u * u;
        double uuu = uu * u;
        double ttt = tt * t;

        Point p = new Point
        {
            X = uuu * Start.X + 3 * uu * t * Control1.X + 3 * u * tt * Control2.X + ttt * End.X,
            Y = uuu * Start.Y + 3 * uu * t * Control1.Y + 3 * u * tt * Control2.Y + ttt * End.Y
        };

        return p;
    }

    // Calculate arrowhead at end point
    protected override void OnRender(DrawingContext dc)
    {
        base.OnRender(dc);

        if (Stroke == null) return;

        // Draw the curve
        var pen = new Pen(Stroke, StrokeThickness);
        pen.Freeze();

        var geometry = DefiningGeometry;
        dc.DrawGeometry(null, pen, geometry);

        // Draw arrowhead
        DrawArrowhead(dc, End, GetAngleAt(0.95));
    }

    private double GetAngleAt(double t)
    {
        // Approximate derivative at t
        double dt = 0.01;
        var p1 = GetPointAt(t);
        var p2 = GetPointAt(t + dt);
        return Math.Atan2(p2.Y - p1.Y, p2.X - p1.X) * 180 / Math.PI;
    }

    private void DrawArrowhead(DrawingContext dc, Point point, double angle)
    {
        double size = 8;
        double arrowAngle = 30 * Math.PI / 180;

        // Calculate arrow points
        Point p1 = point;
        Point p2 = new Point(
            point.X - size * Math.Cos((angle - arrowAngle) * Math.PI / 180),
            point.Y - size * Math.Sin((angle - arrowAngle) * Math.PI / 180));
        Point p3 = new Point(
            point.X - size * Math.Cos((angle + arrowAngle) * Math.PI / 180),
            point.Y - size * Math.Sin((angle + arrowAngle) * Math.PI / 180));

        var arrowGeometry = new PathGeometry();
        var arrowFigure = new PathFigure { StartPoint = p1 };
        arrowFigure.Segments.Add(new LineSegment(p2, true));
        arrowFigure.Segments.Add(new LineSegment(p3, true));
        arrowFigure.IsClosed = true;
        arrowGeometry.Figures.Add(arrowFigure);

        var arrowBrush = Stroke.Clone();
        arrowBrush.Opacity = 0.9;
        var arrowPen = new Pen(arrowBrush, 1);
        arrowPen.Freeze();

        dc.DrawGeometry(arrowBrush, arrowPen, arrowGeometry);
    }
}
