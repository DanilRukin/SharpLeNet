using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.Infrastructure;

public class RelayCommand : ICommand
{
    private readonly Action<object?> _execute;
    private readonly Func<object?, bool>? _canExecute;

    public RelayCommand(Action<object?> execute, Func<object?, bool>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public RelayCommand(Action execute, Func<bool>? canExecute = null)
        : this(_ => execute(), canExecute is null ? null : _ => canExecute())
    {
    }

    public bool CanExecute(object? parameter) => _canExecute?.Invoke(parameter) ?? true;

    public void Execute(object? parameter) => _execute(parameter);

    public event EventHandler? CanExecuteChanged
    {
        add => CommandManager.RequerySuggested += value;
        remove => CommandManager.RequerySuggested -= value;
    }

    public void RaiseCanExecuteChanged() => CommandManager.InvalidateRequerySuggested();
}

public class RelayCommand<T> : ICommand
{
    private readonly Action<T?> _execute;
    private readonly Predicate<T?>? _canExecute;

    public RelayCommand(Action<T?> execute, Predicate<T?>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public RelayCommand(Action execute, Func<bool>? canExecute = null)
        : this(_ => execute(), canExecute is null ? null : _ => canExecute())
    {
    }

    public bool CanExecute(object? parameter)
    {
        if (_canExecute is null)
            return true;

        if (parameter is null && default(T) is null)
            return _canExecute(default);

        if (parameter is T typedParameter)
            return _canExecute(typedParameter);

        return false;
    }

    public void Execute(object? parameter)
    {
        if (parameter is null && default(T) is null)
        {
            _execute(default);
        }
        else if (parameter is T typedParameter)
        {
            _execute(typedParameter);
        }
        else
        {
            _execute(default);
        }
    }

    public event EventHandler? CanExecuteChanged
    {
        add => CommandManager.RequerySuggested += value;
        remove => CommandManager.RequerySuggested -= value;
    }

    public void RaiseCanExecuteChanged() => CommandManager.InvalidateRequerySuggested();
}


