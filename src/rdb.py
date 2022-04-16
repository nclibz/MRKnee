def get_rdb_string():

    import pyodbc
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    params = quote_plus(
        r"Driver={ODBC Driver 18 for SQL Server};Server=tcp:optunasrvr.database.windows.net,1433;Database=optuna;Uid=nclibz;Pwd=MmSrv4G3;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    conn_str = "mssql+pyodbc:///?odbc_connect={}".format(params)
    return conn_str
    # engine_azure = create_engine(conn_str,echo=True)
