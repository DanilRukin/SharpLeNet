using SharpLeNet.Vision.Wpf.Infrastructure;

namespace SharpLeNet.Vision.Wpf.ViewModels;

public class DrawingCanvasViewModel : BaseViewModel
{
    private int _width;
    private int _height;
    private byte[]? _pixelData;
    private bool _hasDrawing;

    public DrawingCanvasViewModel(int width, int height)
    {
        _width = width;
        _height = height;
        _pixelData = new byte[width * height * 4]; // RGBA
        Clear();
    }

    public int Width => _width;
    public int Height => _height;

    public byte[]? PixelData
    {
        get => _pixelData;
        private set => SetProperty(ref _pixelData, value);
    }

    public bool HasDrawing
    {
        get => _hasDrawing;
        private set => SetProperty(ref _hasDrawing, value);
    }

    public void Clear()
    {
        if (_pixelData != null)
        {
            Array.Clear(_pixelData, 0, _pixelData.Length);
        }
        HasDrawing = false;
    }

    public void Save(string filename)
    {
        // Сохранение в файл
    }

    public void AddStroke(int x, int y, int size, byte intensity)
    {
        HasDrawing = true;
        // Здесь логика добавления штриха
    }
}
