import sqlite3

# Connect to the SQLite database file
conn = sqlite3.connect("chinook.db")
cursor = conn.cursor()

# Query to list all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in chinook.db:", tables)

conn.close()
