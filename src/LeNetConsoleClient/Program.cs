using SharpLeNet.Core;
using SharpLeNet.Core.Layers;
using SharpLeNet.Core.Losses;

namespace LeNetConsoleClient
{
    internal class Program
    {
        static void TestLeNetLayers()
        {
            Console.WriteLine("=== Тест слоев LeNet ===");

            // Создаем мини-LeNet архитектуру:
            // Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax

            var model = new Model()
                .AddLayer(new Conv2DLayer(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 2)) // 28x28x1 -> 28x28x6
                .AddLayer(new ReLULayer())
                .AddLayer(new MaxPoolingLayer(poolSize: 2, stride: 2)) // 28x28x6 -> 14x14x6
                .AddLayer(new Conv2DLayer(inputChannels: 6, outputChannels: 16, kernelSize: 5)) // 14x14x6 -> 10x10x16
                .AddLayer(new ReLULayer())
                .AddLayer(new MaxPoolingLayer(poolSize: 2, stride: 2)) // 10x10x16 -> 5x5x16
                .AddLayer(new FlattenLayer()) // 5x5x16 = 400 -> [batch, 400]
                .AddLayer(new LinearLayer(inputSize: 400, outputSize: 120))
                .AddLayer(new ReLULayer())
                .AddLayer(new LinearLayer(inputSize: 120, outputSize: 84))
                .AddLayer(new ReLULayer())
                .AddLayer(new LinearLayer(inputSize: 84, outputSize: 10))
                .AddLayer(new SoftmaxLayer());

            Console.WriteLine($"Количество слоев: {model.Parameters.Count / 2} обучаемых слоев");
            Console.WriteLine($"Всего параметров: {model.Parameters.Sum(p => p.Size)}");

            // Тестовый батч MNIST-like данных (2 изображения 28x28 в градациях серого)
            var input = Tensor.Random(new int[] { 2, 1, 28, 28 });

            Console.WriteLine($"\nВходные данные: [{input.Shape[0]}, {input.Shape[1]}, {input.Shape[2]}, {input.Shape[3]}]");

            // Прямой проход
            var output = model.Forward(input);

            Console.WriteLine($"Выходные данные (10 классов): [{output.Shape[0]}, {output.Shape[1]}]");

            // Проверяем Softmax
            Console.WriteLine("\nПроверка выходов (первые 2 примера):");
            for (int i = 0; i < Math.Min(2, output.Shape[0]); i++)
            {
                Console.Write($"Пример {i}: ");
                double sum = 0;
                for (int j = 0; j < Math.Min(5, output.Shape[1]); j++) // покажем первые 5 классов
                {
                    Console.Write($"{output[i, j]:F4} ");
                    sum += output[i, j];
                }
                if (output.Shape[1] > 5) Console.Write("...");
                Console.WriteLine($" | Сумма: {sum:F6}");
            }

            // Тест autograd через сверточную сеть
            Console.WriteLine("\n=== Тест Autograd через сверточную сеть ===");

            // Создаем случайные метки (one-hot encoding)
            var labels = new Tensor(new int[] { 2, 10 });
            labels[0, 3] = 1; // первый пример: класс 3
            labels[1, 7] = 1; // второй пример: класс 7

            // Вычисляем loss через Softmax+CrossEntropy
            var (softmaxOutput, loss) = output.SoftmaxCrossEntropy(labels);

            Console.WriteLine($"Loss: {loss.Data[0]:F6}");

            // Обнуляем градиенты
            model.ZeroGrad();

            // Обратное распространение
            loss.Backward();

            // Проверяем градиенты
            int paramsWithGrad = model.Parameters.Count(p => p.Grad != null);
            int paramsWithNonZeroGrad = model.Parameters.Count(p =>
                p.Grad != null && p.Grad.Data.Any(v => Math.Abs(v) > 1e-10));

            Console.WriteLine($"Параметры с градиентами: {paramsWithGrad}/{model.Parameters.Count}");
            Console.WriteLine($"Параметры с ненулевыми градиентами: {paramsWithNonZeroGrad}/{model.Parameters.Count}");

            if (paramsWithNonZeroGrad == model.Parameters.Count)
            {
                Console.WriteLine("✓ Все градиенты успешно посчитаны!");
            }
            else
            {
                Console.WriteLine("⚠ Не все градиенты посчитаны правильно!");
            }

            Console.ReadKey();
        }

        static void DebugLinearLayer()
        {
            Console.WriteLine("=== Отладка LinearLayer ===");

            // Минимальный тест
            var layer = new LinearLayer(inputSize: 3, outputSize: 2);

            Console.WriteLine($"Weight shape: [{layer.Weights.Shape[0]}, {layer.Weights.Shape[1]}]");
            Console.WriteLine($"Bias shape: [{layer.Biases.Shape[0]}]");

            // Тестовый вход: 2 примера по 3 признака
            var input = new Tensor(new double[]
            {
                1, 2, 3,   // пример 1
                4, 5, 6    // пример 2
            }, new int[] { 2, 3 });

            Console.WriteLine($"\nInput shape: [{input.Shape[0]}, {input.Shape[1]}]");

            try
            {
                var output = layer.Forward(input);
                Console.WriteLine($"Output shape: [{output.Shape[0]}, {output.Shape[1]}]");
                Console.WriteLine($"Output values: [{output[0, 0]:F4}, {output[0, 1]:F4}; {output[1, 0]:F4}, {output[1, 1]:F4}]");

                // Тест backward
                var loss = output.Sum();
                loss.Backward();

                Console.WriteLine("\nГрадиенты посчитаны успешно!");
                Console.WriteLine($"dL/dWeight ненулевой: {layer.Weights.Grad?.Data.Any(v => Math.Abs(v) > 1e-10)}");
                Console.WriteLine($"dL/dBias ненулевой: {layer.Biases.Grad?.Data.Any(v => Math.Abs(v) > 1e-10)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }


        static void DebugMatMulDirectly()
        {
            Console.WriteLine("=== Прямой тест MatMul ===");

            // A: [2, 3]
            var A = new Tensor(new double[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });

            // B: [3, 2] 
            var B = new Tensor(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, new int[] { 3, 2 });

            try
            {
                var C = A.MatMul(B);
                Console.WriteLine($"Успех! C shape: [{C.Shape[0]}, {C.Shape[1]}]");
                Console.WriteLine($"C[0,0] = {C[0, 0]:F4} (должно быть 1*0.1 + 2*0.3 + 3*0.5 = 2.2)");
                Console.WriteLine($"C[0,1] = {C[0, 1]:F4} (должно быть 1*0.2 + 2*0.4 + 3*0.6 = 2.8)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка: {ex.Message}");
            }
        }

        static void TestLinearGradientsWithDebug()
        {
            Console.WriteLine("=== Тест градиентов LinearLayer с отладкой ===");

            var layer = new LinearLayer(inputSize: 3, outputSize: 2);

            // Фиксируем веса
            layer.Weights.Data[0] = 0.1; layer.Weights.Data[1] = 0.2;
            layer.Weights.Data[2] = 0.3; layer.Weights.Data[3] = 0.4;
            layer.Weights.Data[4] = 0.5; layer.Weights.Data[5] = 0.6;
            layer.Biases.Data[0] = 0.01; layer.Biases.Data[1] = 0.02;

            // Вход требует градиенты
            var input = new Tensor(new double[] { 1, 2, 3 }, new int[] { 1, 3 }, requiresGrad: true);

            Console.WriteLine($"Input requires grad: {input.RequiresGrad}");
            Console.WriteLine($"Weight requires grad: {layer.Weights.RequiresGrad}");
            Console.WriteLine($"Bias requires grad: {layer.Biases.RequiresGrad}");

            // Прямой проход
            var output = layer.Forward(input);
            Console.WriteLine($"\nПрямой проход:");
            Console.WriteLine($"Output shape: [{output.Shape[0]}, {output.Shape[1]}]");
            Console.WriteLine($"Output requires grad: {output.RequiresGrad}");
            Console.WriteLine($"Output: [{output[0, 0]:F4}, {output[0, 1]:F4}]");

            // Проверяем ручной расчет:
            // y0 = 1*0.1 + 2*0.3 + 3*0.5 + 0.01 = 0.1 + 0.6 + 1.5 + 0.01 = 2.21
            // y1 = 1*0.2 + 2*0.4 + 3*0.6 + 0.02 = 0.2 + 0.8 + 1.8 + 0.02 = 2.82
            Console.WriteLine($"Ожидаемо: [2.2100, 2.8200]");

            // Target
            var target = new Tensor(new double[] { 0, 1 }, new int[] { 1, 2 });

            // Loss
            var diff = output - target;
            var diffSq = diff * diff;
            var loss = diffSq.Sum();

            Console.WriteLine($"\nLoss calculation:");
            Console.WriteLine($"diff: [{diff[0, 0]:F4}, {diff[0, 1]:F4}]");
            Console.WriteLine($"diff^2: [{diffSq[0, 0]:F4}, {diffSq[0, 1]:F4}]");
            Console.WriteLine($"loss (sum): {loss.Data[0]:F6}");
            Console.WriteLine($"loss requires grad: {loss.RequiresGrad}");

            // Обнуляем градиенты
            Console.WriteLine("\nОбнуляем градиенты...");
            layer.ZeroGrad();
            if (input.Grad != null) input.Grad.Fill(0);

            // Backward
            Console.WriteLine("Вызываем loss.Backward()...");
            loss.Backward();

            // Проверяем градиенты
            Console.WriteLine("\nГрадиенты:");
            Console.WriteLine($"input.Grad is null: {input.Grad == null}");
            Console.WriteLine($"Weight.Grad is null: {layer.Weights.Grad == null}");
            Console.WriteLine($"Bias.Grad is null: {layer.Biases.Grad == null}");

            if (layer.Weights.Grad != null)
            {
                Console.WriteLine("\ndL/dWeight:");
                Console.WriteLine($"[{layer.Weights.Grad[0, 0]:F6}, {layer.Weights.Grad[0, 1]:F6}]");
                Console.WriteLine($"[{layer.Weights.Grad[1, 0]:F6}, {layer.Weights.Grad[1, 1]:F6}]");
                Console.WriteLine($"[{layer.Weights.Grad[2, 0]:F6}, {layer.Weights.Grad[2, 1]:F6}]");

                Console.WriteLine("\nАналитически:");
                // dL/dy0 = 2*(y0 - t0) = 2*(2.21 - 0) = 4.42
                // dL/dy1 = 2*(y1 - t1) = 2*(2.82 - 1) = 3.64
                // dL/dW = X^T @ dL/dY
                Console.WriteLine($"dL/dW[0,0] = 1*4.42 = 4.420000");
                Console.WriteLine($"dL/dW[0,1] = 1*3.64 = 3.640000");
                Console.WriteLine($"dL/dW[1,0] = 2*4.42 = 8.840000");
                Console.WriteLine($"dL/dW[1,1] = 2*3.64 = 7.280000");
                Console.WriteLine($"dL/dW[2,0] = 3*4.42 = 13.260000");
                Console.WriteLine($"dL/dW[2,1] = 3*3.64 = 10.920000");
            }

            if (layer.Biases.Grad != null)
            {
                Console.WriteLine($"\ndL/dBias: [{layer.Biases.Grad[0]:F6}, {layer.Biases.Grad[1]:F6}]");
                Console.WriteLine($"Аналитически: [4.420000, 3.640000]");
            }
        }

        static void TestSimpleGradient()
        {
            Console.WriteLine("=== Простой тест градиентов ===");

            var layer = new LinearLayer(inputSize: 3, outputSize: 2);

            // Фиксируем веса
            layer.Weights.Data[0] = 0.1; layer.Weights.Data[1] = 0.2;
            layer.Weights.Data[2] = 0.3; layer.Weights.Data[3] = 0.4;
            layer.Weights.Data[4] = 0.5; layer.Weights.Data[5] = 0.6;
            layer.Biases.Data[0] = 0.01; layer.Biases.Data[1] = 0.02;

            // Вход
            var input = new Tensor(new double[] { 1, 2, 3 }, new int[] { 1, 3 }, requiresGrad: true);

            // Прямой проход
            var output = layer.Forward(input);

            // Простая loss: сумма выходов (без квадратов)
            var loss = output.Sum();

            Console.WriteLine($"Output: [{output[0, 0]:F4}, {output[0, 1]:F4}]");
            Console.WriteLine($"Loss: {loss.Data[0]:F4}");

            // Обнуляем градиенты
            layer.ZeroGrad();
            input.ZeroGrad();

            // Backward
            loss.Backward();

            Console.WriteLine("\nГрадиенты:");

            if (layer.Weights.Grad != null)
            {
                Console.WriteLine("dL/dWeights:");
                Console.WriteLine($"[{layer.Weights.Grad[0, 0]:F6}, {layer.Weights.Grad[0, 1]:F6}]");
                Console.WriteLine($"[{layer.Weights.Grad[1, 0]:F6}, {layer.Weights.Grad[1, 1]:F6}]");
                Console.WriteLine($"[{layer.Weights.Grad[2, 0]:F6}, {layer.Weights.Grad[2, 1]:F6}]");
            }

            if (layer.Biases.Grad != null)
            {
                Console.WriteLine($"dL/dBiases: [{layer.Biases.Grad[0]:F6}, {layer.Biases.Grad[1]:F6}]");
            }

            // Аналитический расчет:
            // ∂L/∂output = [1, 1] (так как loss = sum(output))
            // ∂L/∂Weights = input^T @ [1, 1] = [[1, 1], [2, 2], [3, 3]]
            // ∂L/∂Biases = [1, 1]
            Console.WriteLine("\nАналитически ожидается:");
            Console.WriteLine("dL/dWeights:");
            Console.WriteLine("[1.000000, 1.000000]");
            Console.WriteLine("[2.000000, 2.000000]");
            Console.WriteLine("[3.000000, 3.000000]");
            Console.WriteLine("dL/dBiases: [1.000000, 1.000000]");
        }

        static void SimpleTest()
        {
            var a = new Tensor([2.0, 3.0], [2], requiresGrad: true);
            var b = new Tensor([1.0, 1.0], [2]);

            // Простая операция
            var c = a - b;  // [1.0, 2.0]
            var d = c * c;  // [1.0, 4.0]
            var loss = d.Sum();  // 5.0

            loss.Backward();

            // Аналитически:
            // ∂L/∂a = 2*(a-b) = 2*[1.0, 2.0] = [2.0, 4.0]
            Console.WriteLine($"dL/da: [{a.Grad?.Data[0]:F4}, {a.Grad?.Data[1]:F4}]");
        }

        static void TestDivisionAndMSELoss()
        {
            Console.WriteLine("=== Тест деления и MSE Loss ===");

            // Создаем простые тензоры
            var predictions = new Tensor(new double[] { 1.0, 2.0, 3.0, 4.0 }, new int[] { 2, 2 }, requiresGrad: true);
            var targets = new Tensor(new double[] { 0.5, 1.5, 2.5, 3.5 }, new int[] { 2, 2 });

            Console.WriteLine($"Predictions: [{predictions[0, 0]}, {predictions[0, 1]}; {predictions[1, 0]}, {predictions[1, 1]}]");
            Console.WriteLine($"Targets: [{targets[0, 0]}, {targets[0, 1]}; {targets[1, 0]}, {targets[1, 1]}]");

            // Тест деления
            var divResult = predictions / targets;
            Console.WriteLine($"\nДеление predictions / targets:");
            Console.WriteLine($"  [{divResult[0, 0]:F4}, {divResult[0, 1]:F4}; {divResult[1, 0]:F4}, {divResult[1, 1]:F4}]");

            // Тест умножения на скаляр
            var mulScalar = predictions * 2.5;
            Console.WriteLine($"\nУмножение predictions * 2.5:");
            Console.WriteLine($"  [{mulScalar[0, 0]:F4}, {mulScalar[0, 1]:F4}; {mulScalar[1, 0]:F4}, {mulScalar[1, 1]:F4}]");

            // Тест MSE Loss
            var mseLoss = new MSELoss();
            var loss = mseLoss.Compute(predictions, targets);

            Console.WriteLine($"\nMSE Loss: {loss.Data[0]:F6}");

            // Проверка backward
            loss.Backward();

            Console.WriteLine("\nГрадиенты predictions после MSE:");
            if (predictions.Grad != null)
            {
                Console.WriteLine($"  [{predictions.Grad[0, 0]:F6}, {predictions.Grad[0, 1]:F6}; " +
                                 $"{predictions.Grad[1, 0]:F6}, {predictions.Grad[1, 1]:F6}]");

                // Аналитически: dL/dp = 2*(p - t) / N
                // N = 4, p-t = [0.5, 0.5, 0.5, 0.5]
                // 2*(p-t)/N = [0.25, 0.25, 0.25, 0.25]
                Console.WriteLine($"Ожидаемо: [0.250000, 0.250000; 0.250000, 0.250000]");
            }
        }
        static void Main()
        {
            TestDivisionAndMSELoss();
            //SimpleTest();
            //TestSimpleGradient();
            //TestLinearGradientsWithDebug();
            //TestLeNetLayers();
            //DebugLinearLayer();
            //DebugMatMulDirectly();
        }
    }
}
