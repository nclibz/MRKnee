def get_rdb_string():


    !sudo curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
    !sudo mambcurl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list

    !sudo apt-get update
    !sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
    !conda install pyodbc -y

    import pyodbc
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus


    params = quote_plus(r'Driver={ODBC Driver 18 for SQL Server};Server=tcp:optunasrvr.database.windows.net,1433;Database=optuna;Uid=nclibz;Pwd=MmSrv4G3;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;')
    conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
    return conn_str
    #engine_azure = create_engine(conn_str,echo=True)

