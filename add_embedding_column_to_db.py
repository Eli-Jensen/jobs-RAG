import sqlite3

conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

cursor.execute("ALTER TABLE jobs ADD COLUMN embedding BLOB")

conn.commit()
conn.close()