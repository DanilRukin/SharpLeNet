using SharpLeNet.Vision.Wpf.Models;
using System.Collections.ObjectModel;
using System.Windows.Threading;

namespace SharpLeNet.Vision.Wpf.Services;

public class LoggerService : ILoggerService
{
    private readonly ObservableCollection<LogEntry> _logs = new();
    private readonly Dispatcher _dispatcher;
    private readonly int _maxLogEntries = 10000;

    public ObservableCollection<LogEntry> Logs => _logs;

    public event EventHandler<LogEntry>? LogAdded;

    public LoggerService()
    {
        _dispatcher = Dispatcher.CurrentDispatcher;

        // Начальное сообщение
        Info("Logger initialized", "System");
    }

    public void Debug(string message, string? source = null)
    {
        AddLog(new LogEntry(LogLevel.Debug, message, source));
    }

    public void Info(string message, string? source = null)
    {
        AddLog(new LogEntry(LogLevel.Info, message, source));
    }

    public void Warning(string message, string? source = null)
    {
        AddLog(new LogEntry(LogLevel.Warning, message, source));
    }

    public void Error(string message, string? source = null)
    {
        AddLog(new LogEntry(LogLevel.Error, message, source));
    }

    public void Clear()
    {
        _dispatcher.Invoke(() => _logs.Clear());
        Info("Log cleared", "System");
    }

    private void AddLog(LogEntry entry)
    {
        _dispatcher.Invoke(() =>
        {
            _logs.Add(entry);

            // Ограничим количество записей
            if (_logs.Count > _maxLogEntries)
            {
                _logs.RemoveAt(0);
            }
        });

        LogAdded?.Invoke(this, entry);

        // Также пишем в Debug Output
        System.Diagnostics.Debug.WriteLine(entry);
    }
}
