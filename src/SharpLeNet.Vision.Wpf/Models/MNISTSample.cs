namespace SharpLeNet.Vision.Wpf.Models;

public class MNISTSample
{
    public int Index { get; set; }
    public int Label { get; set; }
    public byte[]? ImageData { get; set; }
    public int Width { get; set; } = 28;
    public int Height { get; set; } = 28;
}
