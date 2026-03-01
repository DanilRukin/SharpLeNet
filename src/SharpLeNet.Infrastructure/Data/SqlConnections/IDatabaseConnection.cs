using System.Data;

namespace SharpLeNet.Infrastructure.Data.SqlConnections;

/// <summary>
/// Абстракция для работы с БД (позволяет менять СУБД)
/// </summary>
public interface IDatabaseConnection : IDisposable
{
    void Open();
    void Close();
    int ExecuteNonQuery(string sql, params (string name, object value)[] parameters);
    IEnumerable<T> ExecuteQuery<T>(string sql, Func<IDataReader, T> mapper, params (string name, object value)[] parameters);
    T? ExecuteScalar<T>(string sql, params (string name, object value)[] parameters);
    void BeginTransaction();
    void CommitTransaction();
    void RollbackTransaction();
}
