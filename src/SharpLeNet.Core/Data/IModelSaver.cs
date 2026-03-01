namespace SharpLeNet.Core.Data;

/// <summary>
/// Интерфейс для сохранения моделей
/// </summary>
public interface IModelSaver
{
    /// <summary>
    /// Сохраняет модель
    /// </summary>
    void Save(Model model, string identifier);

    /// <summary>
    /// Загружает модель
    /// </summary>
    Model Load(string identifier);

    /// <summary>
    /// Проверяет, существует ли модель
    /// </summary>
    bool Exists(string identifier);
}
