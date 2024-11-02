import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate and store embeddings
cursor.execute("SELECT id, description FROM jobs")
for job_id, description in cursor.fetchall():
    if job_id % 100 == 0:
        print(f"job_id {job_id}")
    embedding = model.encode(description)
    embedding_blob = np.array(embedding).astype(np.float32).tobytes()

    cursor.execute("UPDATE jobs SET embedding = ? WHERE id = ?", (embedding_blob, job_id))

conn.commit()
conn.close()