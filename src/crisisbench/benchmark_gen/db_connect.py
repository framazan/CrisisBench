import pyodbc
import pandas as pd

# Define the connection parameters
server = 'tcp:vfhistory.database.windows.net,1433'
database = 'vfdata'
username = 'vfhistory'
password = '******'  # replace with your actual password
driver = '{ODBC Driver 18 for SQL Server}'
driver = '{/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.3.so.3.1}'

# Create the connection string
# connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

connectionString = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(connectionString)
cursor = conn.cursor()


cursor.execute("SELECT schema_name FROM information_schema.schemata;")

schemas = cursor.fetchall()
print("Schemas:")
for schema in schemas:
    print(schema[0])


cursor.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE';")

# Fetch and print all table names
tables = cursor.fetchall()
print("Tables:")
for table in tables:
    print(f"Schema: {table[0]}, Table: {table[1]}")
    
# Schema: dbo, Table: feedback (19k rows)
# Schema: dbo, Table: case
# Schema: dbo, Table: agents
# Schema: dbo, Table: messages (3.6M rows)
    
schema_name = 'dbo'
table_name = 'case'

# Query to get all columns of the specified table
sql_query = f"""
    select top 100 * from dbo.messages
    where Ticket_id = 38418000084780834;
"""

df = pd.read_sql(sql_query, conn)
df

df.to_csv("messages.csv")


# Fetch and print all column details
messages = cursor.fetchall()

pd.DataFrame(messages)


print(f"Columns in {schema_name}.{table_name}:")
for column in columns:
    print(f"Name: {column[0]}, Type: {column[1]}, Nullable: {column[2]}, Default: {column[3]}")
