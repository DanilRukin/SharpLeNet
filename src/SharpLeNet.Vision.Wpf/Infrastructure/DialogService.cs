using Microsoft.Win32;
using System.Windows;

namespace SharpLeNet.Vision.Wpf.Infrastructure;

public class DialogService : IDialogService
{
    public void ShowMessage(string message, string title = "Information")
    {
        MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Information);
    }

    public bool ShowConfirmation(string message, string title = "Confirm")
    {
        return MessageBox.Show(message, title, MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes;
    }

    public string? ShowOpenFileDialog(string filter = "All files (*.*)|*.*")
    {
        var dialog = new OpenFileDialog { Filter = filter };
        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }

    public string? ShowSaveFileDialog(string filter = "All files (*.*)|*.*")
    {
        var dialog = new SaveFileDialog { Filter = filter };
        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }
}
