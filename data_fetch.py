import pyodbc
import pandas as pd

# Step 1: Set up the connection
server = 'your_server_name'  # e.g., 'localhost\SQLEXPRESS'
database = 'your_database_name'
username = 'your_username'
password = 'your_password'

# Trusted connection (if using Windows Authentication)
# conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# SQL Auth (if using SQL Server Authentication)
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Step 2: Connect and fetch data
try:
    with pyodbc.connect(conn_str) as conn:
        query = "SELECT * FROM your_table_name"
        df = pd.read_sql(query, conn)

        # Step 3: Export to CSV
        df.to_csv("output.csv", index=False)
        print("Data exported to output.csv successfully.")
except Exception as e:
    print("Error:", e)
