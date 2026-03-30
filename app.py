import os
import time
import random
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError
 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_GROQ_KEY_HERE")
CHROMA_PATH = "./university_data"
COLLECTION_NAME = "gtu_docs"
LLM_MODEL = "llama-3.3-70b-versatile"
N_RETRIEVAL_RESULTS = 10
BASE_SLEEP = 2.0
MAX_RETRIES = 3
 
SYSTEM_PROMPT = """You are the official Academic Registrar assistant for Global Tech University (GTU).
 
STRICT OPERATING RULES:
1. Answer ONLY using information present in the CONTEXT block provided to you.
2. If the context does not contain enough information to answer the question fully,
   state exactly: "The provided documents do not contain sufficient information to
   answer this question. Please contact the Office of the Academic Registrar."
3. Never fabricate course names, prerequisite rules, professor names, or policies.
4. When citing prerequisites, reproduce the exact AND/OR logic as written in the source.
5. Always end your response with a 'Sources:' line listing the document filenames used.
"""
 
def build_context_block(documents: list[str], metadatas: list[dict]) -> tuple[str, list[str]]:
    seen = set()
    sources = []
    for m in metadatas:
        src = m.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
 
    context_lines = []
    for i, doc in enumerate(documents):
        src = metadatas[i].get("source", "unknown")
        context_lines.append(f"[SOURCE: {src}]\n{doc}")
 
    return "\n\n---\n\n".join(context_lines), sources
 
 
def is_context_sufficient(context: str, query: str) -> bool:
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    context_lower = context.lower()
    matched = sum(1 for kw in keywords if kw in context_lower)
    return matched >= max(1, len(keywords) // 3)
 
 
def query_with_retry(groq_client: Groq, model: str, messages: list, retries: int = MAX_RETRIES) -> str:
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content
 
        except RateLimitError:
            wait = BASE_SLEEP * (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"  [RATE LIMIT] Attempt {attempt + 1}/{retries}. Waiting {wait:.1f}s before retry...")
            time.sleep(wait)
 
        except APIConnectionError as e:
            print(f"  [CONNECTION ERROR] {e}. Retrying in {BASE_SLEEP}s...")
            time.sleep(BASE_SLEEP)
 
        except APIStatusError as e:
            print(f"  [API ERROR {e.status_code}] {e.message}")
            break
 
    return "ERROR: Maximum retries exceeded. The query could not be completed."
 
 
def gtu_registrar_agent():
    print("=" * 65)
    print("  GTU ACADEMIC REGISTRAR — RAG QUERY SYSTEM")
    print("  Model  :", LLM_MODEL)
    print("  Index  :", CHROMA_PATH)
    print("=" * 65)
 
    groq_client = Groq(api_key=GROQ_API_KEY)
 
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
 
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        print(f"  Collection loaded: {collection.count()} document chunks indexed.\n")
    except Exception as e:
        print(f"[FATAL] Could not load ChromaDB collection '{COLLECTION_NAME}': {e}")
        print("Ensure your ingestion script has already been run against the './university_data' path.")
        return
 
    test_queries = [
        "List two courses a student can take after completing CS102 with a grade of C",
        "What are the specific prerequisites for CS301?",
        "Suggest a course plan for a CS major who just finished CS102.",
        "Who is the professor for the AI course?",
    ]
 
    print("--- BEGINNING EVALUATION QUERIES ---\n")
 
    for idx, query in enumerate(test_queries, start=1):
        print(f"QUERY {idx}/{len(test_queries)}: {query}")
 
        retrieval = collection.query(
            query_texts=[query],
            n_results=N_RETRIEVAL_RESULTS
        )
 
        raw_docs = retrieval["documents"][0]
        raw_meta = retrieval["metadatas"][0]
 
        context_block, sources = build_context_block(raw_docs, raw_meta)
 
        if not is_context_sufficient(context_block, query):
            print("  [CONTEXT CHECK] Low relevance signal detected. Proceeding with available context.")
 
        user_message = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"Remember: answer using only the context above. "
            f"If the answer is not present, say so explicitly."
        )
 
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
 
        sleep_duration = BASE_SLEEP + random.uniform(0.2, 0.8)
        time.sleep(sleep_duration)
 
        answer = query_with_retry(groq_client, LLM_MODEL, messages)
 
        print(f"\n  RETRIEVED SOURCES : {sources}")
        print(f"\n  ANSWER:\n{answer}")
        print("\n" + "-" * 65 + "\n")
 
    print("--- EVALUATION COMPLETE ---")
 
 
if __name__ == "__main__":
    gtu_registrar_agent()
