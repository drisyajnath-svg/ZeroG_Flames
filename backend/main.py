import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# 1. Configure Gemini API (Paste your key here)
GEMINI_API_KEY="AQ.Ab8RN6J_AhTgfJTEkr97pIM2_g9HQ8cufOiSkWGC3iSlQdTIsQ"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# 2. Initialize FastAPI + CORS for Drisya's Frontend
app = FastAPI(title="ZeroG Flames Fire Safety API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,  # <-- Change this to False
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Connect to your existing Vector DB
DB_PATH = r"D:\NASA\ZeroG_Flames\nasa_vectordb"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(
    name="combustion_experiments",
    embedding_function=sentence_transformer_ef
)

# Request format expected from Drisya's frontend
class QueryRequest(BaseModel):
    query: str
    environment: str = "General"  # e.g., "ISS (0g)", "Moon (1/6g)", "Mars (3/8g)"

SYSTEM_PROMPT = """You are the backend AI Engine for a Spaceflight Fire Safety Dashboard, specializing in microgravity and partial-gravity combustion. Analyze raw text chunks retrieved from NASA's Physical Sciences Informatics (PSI) database and answer user queries.

DOMAIN RULES:
- Evaluate fire safety ONLY for habitable crew environments (spacecraft, Moon/Mars bases, rovers).
- IGNORE any data related to rocket engine propellants.
- Categorize gravity regimes clearly: Microgravity (ISS/Orbit), Lunar (1/6 g), and Martian (3/8 g). Note that partial-gravity data is bounded by 0g and 1g results or mimicked via electric fields (e.g., E-FIELD Flames).

REAL-TIME PROCESSING:
Before answering, generate a thinking process wrapped in <PROCESSING_LOG> tags identifying:
1. User intent & target environment
2. Specific NASA investigations found in context (e.g., ACME, BASS, FLEX, Saffire)
3. Key variables extracted (oxygen %, pressure, fuel type, airflow speed)
4. Missing data or caveats

STRICT JSON OUTPUT:
After </PROCESSING_LOG>, output ONLY a raw, valid JSON object (no ```json markdown fences):
{{
  "summary": "2-3 sentence direct answer based ONLY on the provided context.",
  "key_variables": {{
    "fuel_type": "...",
    "oxygen_concentration": "...",
    "gravity_level": "..."
  }},
  "comparisons": "How this compares to Earth (1g) or other investigations.",
  "terrestrial_applications": "Earth-based fire safety applications if mentioned, else null.",
  "confidence_warning": "What is missing or bounded if context is incomplete, else null."
}}

CONTEXT FROM NASA DATABASE:
{context}

TARGET ENVIRONMENT FILTER: {environment}
USER QUERY: {query}
"""

@app.post("/ask_dashboard")
async def ask_dashboard(request: QueryRequest):
    try:
        # Step A: Retrieve top 4 relevant chunks from ChromaDB
        search_query = f"{request.environment} {request.query}"
        results = collection.query(
            query_texts=[search_query],
            n_results=4
        )
        
        retrieved_chunks = results["documents"][0]
        context_text = "\n\n---\n\n".join(retrieved_chunks)

        # Step B: Send context + query to Gemini
        prompt = SYSTEM_PROMPT.format(
            context=context_text,
            environment=request.environment,
            query=request.query
        )
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Step C: Split real-time log and JSON payload
        if "</PROCESSING_LOG>" in raw_text:
            parts = raw_text.split("</PROCESSING_LOG>")
            log_part = parts[0].replace("<PROCESSING_LOG>", "").strip()
            json_part = parts[1].strip()
            print(f"\n=== REAL-TIME AI LOG ===\n{log_part}\n========================\n")
        else:
            json_part = raw_text

        # Clean accidental markdown fences if Gemini adds them
        if json_part.startswith("```"):
            json_part = json_part.strip("`").replace("json\n", "", 1).strip()

        parsed_data = json.loads(json_part)

        return {
            "status": "success",
            "data": parsed_data,
            "sources": retrieved_chunks  
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))