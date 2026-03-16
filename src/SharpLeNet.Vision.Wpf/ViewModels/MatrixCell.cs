using SharpLeNet.Vision.Wpf.Infrastructure;
using System.Windows.Media;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class MatrixCell : BaseViewModel
{
    private int _row;
    private int _column;
    private int _value;
    private Brush _color;
    private Brush _textColor;
    private string _toolTip;

    public int Row
    {
        get => _row;
        set => SetProperty(ref _row, value);
    }

    public int Column
    {
        get => _column;
        set => SetProperty(ref _column, value);
    }

    public int Value
    {
        get => _value;
        set => SetProperty(ref _value, value);
    }

    public Brush Color
    {
        get => _color;
        set => SetProperty(ref _color, value);
    }

    public Brush TextColor
    {
        get => _textColor;
        set => SetProperty(ref _textColor, value);
    }

    public string ToolTip
    {
        get => _toolTip;
        set => SetProperty(ref _toolTip, value);
    }
}
