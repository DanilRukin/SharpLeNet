using System.Windows;
using System.Windows.Input;

namespace SharpLeNet.Vision.Wpf.Behaviors;

public static class DropBehavior
{
    public static readonly DependencyProperty IsDropTargetProperty =
        DependencyProperty.RegisterAttached("IsDropTarget", typeof(bool), typeof(DropBehavior),
            new PropertyMetadata(false, OnIsDropTargetChanged));

    public static readonly DependencyProperty DropFormatProperty =
        DependencyProperty.RegisterAttached("DropFormat", typeof(string), typeof(DropBehavior),
            new PropertyMetadata(null));

    public static readonly DependencyProperty DropCommandProperty =
        DependencyProperty.RegisterAttached("DropCommand", typeof(ICommand), typeof(DropBehavior),
            new PropertyMetadata(null));

    public static readonly DependencyProperty DropPositionProperty =
        DependencyProperty.RegisterAttached("DropPosition", typeof(Point), typeof(DropBehavior),
            new PropertyMetadata(default(Point)));

    public static void SetIsDropTarget(UIElement element, bool value) =>
        element.SetValue(IsDropTargetProperty, value);

    public static bool GetIsDropTarget(UIElement element) =>
        (bool)element.GetValue(IsDropTargetProperty);

    public static void SetDropFormat(UIElement element, string value) =>
        element.SetValue(DropFormatProperty, value);

    public static string GetDropFormat(UIElement element) =>
        (string)element.GetValue(DropFormatProperty);

    public static void SetDropCommand(UIElement element, ICommand value) =>
        element.SetValue(DropCommandProperty, value);

    public static ICommand GetDropCommand(UIElement element) =>
        (ICommand)element.GetValue(DropCommandProperty);

    public static void SetDropPosition(UIElement element, Point value) =>
        element.SetValue(DropPositionProperty, value);

    public static Point GetDropPosition(UIElement element) =>
        (Point)element.GetValue(DropPositionProperty);

    private static void OnIsDropTargetChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is UIElement element)
        {
            if ((bool)e.NewValue)
            {
                element.AllowDrop = true;
                element.DragEnter += OnDragEnter;
                element.DragOver += OnDragOver;
                element.DragLeave += OnDragLeave;
                element.Drop += OnDrop;
            }
            else
            {
                element.AllowDrop = false;
                element.DragEnter -= OnDragEnter;
                element.DragOver -= OnDragOver;
                element.DragLeave -= OnDragLeave;
                element.Drop -= OnDrop;
            }
        }
    }

    private static void OnDragEnter(object sender, DragEventArgs e)
    {
        if (sender is FrameworkElement element)
        {
            var format = GetDropFormat(element);
            if (e.Data.GetDataPresent(format))
            {
                e.Effects = DragDropEffects.Copy;
                element.SetValue(DropPositionProperty, e.GetPosition(element));
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }
            e.Handled = true;
        }
    }

    private static void OnDragOver(object sender, DragEventArgs e)
    {
        if (sender is FrameworkElement element)
        {
            var format = GetDropFormat(element);
            if (e.Data.GetDataPresent(format))
            {
                e.Effects = DragDropEffects.Copy;
                element.SetValue(DropPositionProperty, e.GetPosition(element));
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }
            e.Handled = true;
        }
    }

    private static void OnDragLeave(object sender, DragEventArgs e)
    {
        e.Handled = true;
    }

    private static void OnDrop(object sender, DragEventArgs e)
    {
        if (sender is FrameworkElement element)
        {
            var format = GetDropFormat(element);
            var command = GetDropCommand(element);

            if (command != null && e.Data.GetDataPresent(format))
            {
                var data = e.Data.GetData(format);
                var position = e.GetPosition(element);

                // Snap to grid (8px)
                position.X = Math.Floor(position.X / 8) * 8;
                position.Y = Math.Floor(position.Y / 8) * 8;

                element.SetValue(DropPositionProperty, position);

                if (command.CanExecute(data))
                {
                    command.Execute(data);
                }
            }
            e.Handled = true;
        }
    }
}
