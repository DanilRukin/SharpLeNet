namespace SharpLeNet.Vision.Wpf.Models;

public class LogEntry
{
    public DateTime Timestamp { get; set; }
    public LogLevel Level { get; set; }
    public string Message { get; set; } = string.Empty;
    public string? Source { get; set; }

    public LogEntry() { }

    public LogEntry(LogLevel level, string message, string? source = null)
    {
        Timestamp = DateTime.Now;
        Level = level;
        Message = message;
        Source = source;
    }

    public override string ToString()
    {
        return $"[{Timestamp:HH:mm:ss.fff}] [{Level}] {Message}";
    }
}
