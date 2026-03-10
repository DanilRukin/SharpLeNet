using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.Windows;

namespace SharpLeNet.Vision.Wpf
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // Проверим, какие ресурсы доступны
            var resources = this.Resources.MergedDictionaries;
            foreach (var dict in resources)
            {
                Debug.WriteLine($"Loaded resource dictionary: {dict.Source}");

                foreach (var key in dict.Keys)
                {
                    Debug.WriteLine($"  - {key}: {dict[key]?.GetType().Name}");
                }
            }

            // Проверим конкретные ресурсы
            Debug.WriteLine($"BorderLightBrush: {this.Resources["BorderLightBrush"]?.GetType().Name ?? "NULL"}");
            Debug.WriteLine($"MainBackgroundBrush: {this.Resources["MainBackgroundBrush"]?.GetType().Name ?? "NULL"}");
            Debug.WriteLine($"White80Brush: {this.Resources["White80Brush"]?.GetType().Name ?? "NULL"}");
        }
    }

}
