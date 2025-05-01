import sqlite3

# Step 1: Connect to an SQLite database (it will create the file if it doesn't exist)
conn = sqlite3.connect('chinook.db')  # This creates the 'chinook.db' file in the current directory
cursor = conn.cursor()

# Step 2: Open and read the Chinook.sql file
with open('Chinook.sql', 'r', encoding='utf-8') as sql_file:
    sql_script = sql_file.read()

# Step 3: Execute the SQL script (this will create the tables and insert data)
cursor.executescript(sql_script)

# Step 4: Commit changes and close the connection
conn.commit()
conn.close()

print("Database setup complete! The SQLite database 'chinook.db' has been created.")
