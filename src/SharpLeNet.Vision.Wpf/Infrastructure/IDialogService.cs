namespace SharpLeNet.Vision.Wpf.Infrastructure;

public interface IDialogService
{
    void ShowMessage(string message, string title = "Information");
    bool ShowConfirmation(string message, string title = "Confirm");
    string? ShowOpenFileDialog(string filter = "All files (*.*)|*.*");
    string? ShowSaveFileDialog(string filter = "All files (*.*)|*.*");
}
