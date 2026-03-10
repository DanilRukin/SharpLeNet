using System.Windows;

namespace SharpLeNet.Vision.Wpf.Behaviors;

public static class DragDropBehavior
{
    public static readonly DependencyProperty IsDragSourceProperty =
        DependencyProperty.RegisterAttached("IsDragSource", typeof(bool), typeof(DragDropBehavior),
            new PropertyMetadata(false, OnIsDragSourceChanged));

    public static readonly DependencyProperty DragDataProperty =
        DependencyProperty.RegisterAttached("DragData", typeof(object), typeof(DragDropBehavior),
            new PropertyMetadata(null));

    public static readonly DependencyProperty DragFormatProperty =
        DependencyProperty.RegisterAttached("DragFormat", typeof(string), typeof(DragDropBehavior),
            new PropertyMetadata(null));

    public static void SetIsDragSource(UIElement element, bool value) =>
        element.SetValue(IsDragSourceProperty, value);

    public static bool GetIsDragSource(UIElement element) =>
        (bool)element.GetValue(IsDragSourceProperty);

    public static void SetDragData(UIElement element, object value) =>
        element.SetValue(DragDataProperty, value);

    public static object GetDragData(UIElement element) =>
        element.GetValue(DragDataProperty);

    public static void SetDragFormat(UIElement element, string value) =>
        element.SetValue(DragFormatProperty, value);

    public static string GetDragFormat(UIElement element) =>
        (string)element.GetValue(DragFormatProperty);

    private static void OnIsDragSourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is UIElement element)
        {
            if ((bool)e.NewValue)
            {
                element.PreviewMouseLeftButtonDown += OnPreviewMouseLeftButtonDown;
            }
            else
            {
                element.PreviewMouseLeftButtonDown -= OnPreviewMouseLeftButtonDown;
            }
        }
    }

    private static void OnPreviewMouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (sender is UIElement element && GetIsDragSource(element))
        {
            var data = GetDragData(element);
            var format = GetDragFormat(element) ?? "Unknown";

            if (data != null)
            {
                DragDrop.DoDragDrop(element, new DataObject(format, data), DragDropEffects.Copy);
            }
        }
    }
}
