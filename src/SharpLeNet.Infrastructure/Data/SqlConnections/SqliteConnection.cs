using Microsoft.Data.Sqlite;
using System.Data;

namespace SharpLeNet.Infrastructure.Data.SqlConnections;

/// <summary>
/// Реализация для SQLite
/// </summary>
public class SqliteConnection : IDatabaseConnection
{
    private readonly string _connectionString;
    private SqliteConnection? _connection;

    public SqliteConnection(string databasePath)
    {
        _connectionString = $"Data Source={databasePath}";
    }

    public void Open()
    {
        _connection = new SqliteConnection(_connectionString);
        _connection.Open();
    }

    public void Close()
    {
        _connection?.Close();
    }

    public int ExecuteNonQuery(string sql, params (string name, object value)[] parameters)
    {
        using var command = CreateCommand(sql, parameters);
        return command.ExecuteNonQuery();
    }

    public IEnumerable<T> ExecuteQuery<T>(string sql, Func<IDataReader, T> mapper, params (string name, object value)[] parameters)
    {
        using var command = CreateCommand(sql, parameters);
        using var reader = command.ExecuteReader();

        while (reader.Read())
        {
            yield return mapper(reader);
        }
    }

    public T? ExecuteScalar<T>(string sql, params (string name, object value)[] parameters)
    {
        using var command = CreateCommand(sql, parameters);
        var result = command.ExecuteScalar();
        return result == DBNull.Value ? default : (T)result;
    }

    public void BeginTransaction()
    {
        _connection?.BeginTransaction();
    }

    public void CommitTransaction()
    {
        _connection?.CommitTransaction();
    }

    public void RollbackTransaction()
    {
        _connection?.RollbackTransaction();
    }

    private SqliteCommand CreateCommand(string sql, params (string name, object value)[] parameters)
    {
        var command = _connection!.CreateCommand(sql, parameters);
        command.CommandText = sql;

        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue($"@{name}", value);
        }

        return command;
    }

    public void Dispose()
    {
        _connection?.Dispose();
    }
}
