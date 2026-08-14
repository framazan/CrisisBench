import pyodbc

# Define the connection parameters
server = 'tcp:vfhistory.database.windows.net,1433'
database = 'vfdata'
username = 'vfhistory'
password = ''  # replace with your actual password
driver = '{ODBC Driver 18 for SQL Server}'
driver = '{/opt/homebrew/lib/libmsodbcsql.18.dylib}'

# Create the connection string
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

connectionString = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(connectionString)

# Connect to the database
try:
    conn = pyodbc.connect(connection_string)
    print("Connection successful!")
    
    # Example query
    cursor = conn.cursor()
    cursor.execute("SELECT @@version;")
    
    row = cursor.fetchone()
    while row:
        print(row[0])
        row = cursor.fetchone()
    
    # Close the connection
    conn.close()

except Exception as e:
    print(f"Error: {e}")
