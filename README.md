# GTU-Registrar-RAG-Assistant
# 🎓 GTU Academic Registrar — Agentic RAG Assistant

A high-precision Retrieval-Augmented Generation (RAG) system built to navigate complex university catalogs, prerequisite chains (AND/OR logic), and academic policies with 100% grounded accuracy.

## 🚀 Key Features
- **Deterministic Logic:** Powered by **Groq (Llama-3.3-70b)** with Temperature 0.0 to ensure zero-hallucination policy enforcement.
- **Context-Aware Retrieval:** Utilizes a **ChromaDB** vector store with a K-10 retrieval window to capture multi-hop prerequisite dependencies.
- **Academic Integrity:** Strictly adheres to provided context; includes verifiable citations for every claim.

## 📁 Project Structure
- `university_data/`: Synthetic catalog containing 33 documents (~30k words).
- `ingestion.py`: Script to build the local ChromaDB vector index from raw text.
- `app.py`: The main RAG Agent that processes student queries and generates cited responses.
- `requirements.txt`: Project dependencies for environment setup.
- `evaluation_results.txt`: Results of a 25-query stress test covering eligibility and policy.
- `GTU_Final_Report.pdf`: Technical write-up covering architecture and data strategy.

## 🛠️ Setup & Installation
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/GTU-Registrar-RAG-Assistant.git](https://github.com/YOUR_USERNAME/GTU-Registrar-RAG-Assistant.git)

## Install Dependencies:
Bash
pip install -r requirements.txt

## Environment Setup:
Obtain a Groq API Key and add it to the GROQ_API_KEY variable in app.py.

## Run the System:
First, run python ingestion.py to index the catalog.
Then, run python app.py to start the assistant.

## 📊 Performance Metrics
Citation Coverage: 100%
Hallucination Rate: 0%
Logic Handling: Successfully parses complex "AND/OR" prerequisites and "Minimum Grade" requirements.
