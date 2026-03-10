using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class ConnectionViewModel : BaseViewModel
{
    private Point _startPoint;
    private Point _endPoint;
    private Point _controlPoint1;
    private Point _controlPoint2;

    public string SourceId { get; set; } = string.Empty;
    public string TargetId { get; set; } = string.Empty;
    public LayerBlockViewModel SourceBlock { get; set; } = null!;
    public LayerBlockViewModel TargetBlock { get; set; } = null!;

    public Point StartPoint
    {
        get => _startPoint;
        private set => SetProperty(ref _startPoint, value);
    }

    public Point EndPoint
    {
        get => _endPoint;
        private set => SetProperty(ref _endPoint, value);
    }

    public Point ControlPoint1
    {
        get => _controlPoint1;
        private set => SetProperty(ref _controlPoint1, value);
    }

    public Point ControlPoint2
    {
        get => _controlPoint2;
        private set => SetProperty(ref _controlPoint2, value);
    }

    public void UpdatePath()
    {
        // Calculate connection points based on block positions
        StartPoint = new Point(SourceBlock.Position.X + 210, SourceBlock.Position.Y + 60); // Right side of source
        EndPoint = new Point(TargetBlock.Position.X, TargetBlock.Position.Y + 60); // Left side of target

        // Calculate bezier control points
        var dx = EndPoint.X - StartPoint.X;
        var offset = Math.Min(100, dx / 2);

        ControlPoint1 = new Point(StartPoint.X + offset, StartPoint.Y);
        ControlPoint2 = new Point(EndPoint.X - offset, EndPoint.Y);
    }
}
