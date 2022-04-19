from urllib.parse import quote_plus


def get_rdb_string() -> str:
    srv_str = quote_plus(
        r'Driver={ODBC Driver 18 for SQL Server};Server=tcp:optunasrvr.database.windows.net,1433;Database=optuna;Uid=nclibz;Pwd=MmSrv4G3;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    )
    con_str = f'mssql+pyodbc:///?odbc_connect={srv_str}'
    return con_str
