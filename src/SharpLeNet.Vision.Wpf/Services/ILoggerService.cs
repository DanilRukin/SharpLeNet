using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;

namespace SharpLeNet.Vision.Wpf.Services;

public interface ILoggerService
{
    ObservableCollection<LogEntry> Logs { get; }

    void Debug(string message, string? source = null);
    void Info(string message, string? source = null);
    void Warning(string message, string? source = null);
    void Error(string message, string? source = null);
    void Clear();
    event EventHandler<LogEntry>? LogAdded;
}
