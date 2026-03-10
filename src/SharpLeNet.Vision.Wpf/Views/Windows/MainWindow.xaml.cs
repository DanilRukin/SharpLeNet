using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SharpLeNet.Vision.Wpf.Views.Windows
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // Находим элементы управления в шаблоне
            if (Template.FindName("PART_TitleBar", this) is Border titleBar)
            {
                titleBar.MouseLeftButtonDown += TitleBar_MouseLeftButtonDown;
            }

            if (Template.FindName("PART_DragArea", this) is Grid dragArea)
            {
                dragArea.MouseLeftButtonDown += TitleBar_MouseLeftButtonDown;
            }

            if (Template.FindName("PART_MinimizeButton", this) is Button minimizeButton)
            {
                minimizeButton.Click += MinimizeButton_Click;
            }

            if (Template.FindName("PART_MaximizeButton", this) is Button maximizeButton)
            {
                maximizeButton.Click += MaximizeButton_Click;
            }

            if (Template.FindName("PART_CloseButton", this) is Button closeButton)
            {
                closeButton.Click += CloseButton_Click;
            }
        }

        private void TitleBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ClickCount == 2)
            {
                // Двойной клик - развернуть/восстановить
                WindowState = WindowState == WindowState.Maximized
                    ? WindowState.Normal
                    : WindowState.Maximized;
                return;
            }

            // Одинарный клик - перетаскивание
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                try
                {
                    // Если окно максимизировано, сначала восстанавливаем его
                    if (WindowState == WindowState.Maximized)
                    {
                        var point = PointToScreen(e.GetPosition(this));
                        WindowState = WindowState.Normal;

                        // Позиционируем окно так, чтобы курсор оставался на заголовке
                        this.Left = point.X - (this.Width / 2);
                        this.Top = point.Y - 10;
                    }

                    // Начинаем перетаскивание
                    this.DragMove();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"DragMove error: {ex.Message}");
                }
            }
        }

        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }

        private void MaximizeButton_Click(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState == WindowState.Maximized
                ? WindowState.Normal
                : WindowState.Maximized;
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}