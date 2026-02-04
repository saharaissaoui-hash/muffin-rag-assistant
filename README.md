# muffin-rag-assistant

A French Retrieval-Augmented Generation (RAG) application that recommends **only muffin recipes** based on user ingredients.  
The system enforces a strict muffin-only constraint and generates responses in French using a local LLM.

This project was built as part of an NLP course at **Mines Paris**.

---

## Project Overview

The goal is to build a complete RAG pipeline:

1. Load and clean a recipe dataset
2. Keep only muffin recipes
3. Translate them into French
4. Build embeddings and a vector database
5. Retrieve relevant muffins for a user query
6. Generate a response using a local LLM
7. Provide a simple interactive UI

The assistant persona is **Chef Muffin**, a friendly but obsessive muffin specialist who refuses any non-muffin request.

---

## Architecture

```
User query (French)
        ↓
Embedding (multilingual MiniLM)
        ↓
Chroma vector search (top_k muffins)
        ↓
Context building
        ↓
Local LLM (Mistral via Ollama)
        ↓
French muffin-only response
```

---

## Repository Structure

```
muffin-rag-assistant/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── data/
│   ├── raw/                    # Raw datasets (ignored by git)
│   ├── processed/              # Clean muffin datasets
│   └── vector_store/           # Chroma persistent DB
│
├── src/
│   ├── ingestion/
│   │   ├── inspect_hf_dataset.py
│   │   ├── extract_muffins.py
│   │   └── translate_muffins_to_french.py
│   │
│   ├── embeddings/
│   │   └── build_vector_store.py
│   │
│   ├── retrieval/
│   │   └── retrieve.py
│   │
│   └── generation/
│       ├── chef_muffin.py
│       └── ollama_client.py
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.12** recommended
- macOS or Linux
- Ollama installed locally

---

## Installation

### 1. Clone the repository

```bash
git clone <https://github.com/saharaissaoui-hash/muffin-rag-assistant.git>
cd muffin-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install and Run the Local LLM (Ollama)

### Install Ollama

Download from:

```
https://ollama.com
```

Or via Homebrew:

```bash
brew install ollama
```

### Start the server

```bash
ollama serve
```

Leave this terminal running.

### Download the model

In a new terminal:

```bash
ollama pull mistral
```

Test it:

```bash
ollama run mistral
```

---

## Data Pipeline (ETL)

### Step 1 — Extract only muffins

```bash
python -m src.ingestion.extract_muffins
```

Output:

```
data/processed/muffins_only.csv
```

### Step 2 — Translate muffins to French

```bash
python -m src.ingestion.translate_muffins_to_french
```

Output:

```
data/processed/muffins_only_fr.csv
```

---

## Build the Vector Store

This creates the persistent Chroma database.

```bash
python -m src.embeddings.build_vector_store --rebuild
```

Output:

```
data/vector_store/chroma/
```

---

## Run the Application

Make sure Ollama is running:

```bash
ollama serve
```

Then launch the UI:

```bash
streamlit run app/streamlit_app.py
```

Open the browser at:

```
http://localhost:8501
```

---

## Example Queries

Try:

```
Je veux un muffin au chocolat
J'ai des bananes et du yaourt
Je veux un muffin salé avec du fromage
```

If you ask for anything else:

```
Je veux une pizza
```

The assistant will refuse and redirect to muffins.

---

## Key Design Decisions

### 1. French-only generation
Even though the original dataset is English, all muffins are translated to French before embedding and retrieval.

### 2. Deterministic guardrail
Non-muffin queries are blocked at the code level before generation.

### 3. Local LLM
The system uses a local Mistral model via Ollama:
- no API keys
- no cost
- fully reproducible offline

### 4. Persistent vector database
Embeddings are stored in Chroma and reused across sessions.

---

## Technologies Used

- Python
- Hugging Face Datasets
- SentenceTransformers
- ChromaDB
- Ollama (local Mistral LLM)
- Streamlit

---

## How to Reproduce from Scratch

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. start ollama
ollama serve

# 3. pull model
ollama pull mistral

# 4. build dataset
python -m src.ingestion.extract_muffins
python -m src.ingestion.translate_muffins_to_french

# 5. build vector store
python -m src.embeddings.build_vector_store --rebuild

# 6. run UI
streamlit run app/streamlit_app.py
```

---

## Limitations

- Small muffin dataset (~89 recipes)
- Translation quality depends on the MT model
- Local LLM generation speed depends on hardware

---

## Future Improvements

- Larger French recipe dataset
- Ingredient-aware retrieval
- Evaluation metrics for RAG quality
- Dockerized deployment

---

## Author

Sahar Aissaoui  
Engineering Student at Mines Paris-PSL 
NLP / Applied AI Project
