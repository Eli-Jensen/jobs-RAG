import re
import sqlite3
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

app = FastAPI()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = pipeline("text-generation", model="gpt2")

conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# FAISS index
embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

# Load embeddings from the database into FAISS
cursor.execute("SELECT id, embedding FROM jobs WHERE embedding IS NOT NULL")
id_to_data = {}
for job_id, embedding_blob in cursor.fetchall():
    # Convert blob back to numpy array
    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    index.add(np.array([embedding]))
    
    id_to_data[job_id] = cursor.execute(
        "SELECT title, company_name, location, url, description FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()

def filter_jobs_by_keywords(jobs, keywords):
    """Filter jobs based on the presence of keywords in the description."""
    keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
    filtered_jobs = [
        job for job in jobs if re.search(keyword_pattern, job['description'], re.IGNORECASE)
    ]
    return filtered_jobs if filtered_jobs else jobs

@app.get("/recommend_jobs")
async def recommend_jobs(query: str):
    query_embedding = embedding_model.encode(query).astype(np.float32)
    
    # Search for the closest embeddings in FAISS
    k = 10
    _, indices = index.search(np.array([query_embedding]), k)
    
    # Retrieve job data based on FAISS results
    closest_jobs = []
    for idx in indices[0]:
        job_id = list(id_to_data.keys())[idx]
        job_data = id_to_data[job_id]
        closest_jobs.append({
            "title": job_data[0],
            "company_name": job_data[1],
            "location": job_data[2],
            "url": job_data[3],
            "description": job_data[4]
        })

    # Extract keywords from the user's query
    keywords = [word.strip() for word in re.split(r'[,\s]+', query) if len(word) > 2]

    # Prioritize jobs matching specific keywords
    filtered_jobs = filter_jobs_by_keywords(closest_jobs, keywords)
    llm_input = "\n".join([f"{job['title']} at {job['company_name']} in {job['location']}" for job in filtered_jobs])
    llm_input = f"Find relevant job postings:\n{llm_input}"

    # Generate a response using the local LLM
    llm_response = llm(llm_input, truncation=True, max_new_tokens=50)[0]["generated_text"]

    # Remove descriptions from final response
    results = [
        {key: value for key, value in job.items() if key != "description"}
        for job in filtered_jobs
    ]

    return {"recommendations": results, "llm_response": llm_response}
