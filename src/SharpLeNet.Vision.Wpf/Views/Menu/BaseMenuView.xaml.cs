using SharpLeNet.Vision.Wpf.ViewModels.Menu;
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

namespace SharpLeNet.Vision.Wpf.Views.Menu
{
    /// <summary>
    /// Логика взаимодействия для BaseMenuView.xaml
    /// </summary>
    public partial class BaseMenuView : UserControl
    {
        public BaseMenuView()
        {
            InitializeComponent();
        }

        private void Background_MouseDown(object sender, MouseButtonEventArgs e)
        {
            // Закрываем меню при клике на фон
            var vm = DataContext as BaseMenuViewModel;
            vm?.CloseCommand.Execute(null);
        }
    }
}
