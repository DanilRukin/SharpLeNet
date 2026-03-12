using SharpLeNet.Vision.Wpf.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SharpLeNet.Vision.Wpf.Views.Controls
{
    /// <summary>
    /// Логика взаимодействия для ArchitectureCanvas.xaml
    /// </summary>
    public partial class ArchitectureCanvas : UserControl
    {
        private Point _lastMousePosition;
        private bool _isDragging;
        private LayerBlock? _draggedBlock;
        private ConnectionLine? _tempConnection;

        public ArchitectureCanvas()
        {
            InitializeComponent();
            _tempConnection = TempConnection;
        }

        private ArchitectureViewModel? ViewModel => DataContext as ArchitectureViewModel;

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (ViewModel == null) return;

            var position = e.GetPosition(sender as IInputElement);
            ViewModel.MousePosition = position;

            // Update temporary connection if dragging between blocks
            if (_isDragging && _draggedBlock != null)
            {
                _tempConnection!.Start = new Point(
                    _draggedBlock.RenderTransform.Value.OffsetX + _draggedBlock.ActualWidth,
                    _draggedBlock.RenderTransform.Value.OffsetY + _draggedBlock.ActualHeight / 2);
                _tempConnection.End = position;
                _tempConnection.Visibility = Visibility.Visible;
            }
        }

        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _lastMousePosition = e.GetPosition(sender as IInputElement);
        }

        public void StartConnectionDrag(LayerBlock sourceBlock)
        {
            _isDragging = true;
            _draggedBlock = sourceBlock;
            _tempConnection!.Visibility = Visibility.Visible;
        }

        public void EndConnectionDrag(LayerBlock? targetBlock)
        {
            if (_isDragging && _draggedBlock != null && targetBlock != null && ViewModel != null)
            {
                var connection = new Tuple<LayerBlockViewModel, LayerBlockViewModel>(
                    _draggedBlock.DataContext as LayerBlockViewModel,
                    targetBlock.DataContext as LayerBlockViewModel);

                ViewModel.ConnectLayersCommand.Execute(connection);
            }

            _isDragging = false;
            _draggedBlock = null;
            _tempConnection!.Visibility = Visibility.Collapsed;
        }
    }
}
