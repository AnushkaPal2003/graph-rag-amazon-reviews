# 📱 Graph RAG — Amazon Phone Reviews
### Built with: Neo4j + Azure OpenAI + Sentence Transformers

---

## 🤔 What is this project?

This project lets you ask questions like:
- **"Which is the best phone to buy?"**
- **"Compare Apple vs Samsung"**
- **"Any phone with heating issues?"**
- **"Which brand has the best rating?"**

And get **intelligent, data-driven answers** powered by a Knowledge Graph + AI.

This is called **Graph RAG** — Retrieval Augmented Generation using a Graph Database.

---

## 🧠 How does Graph RAG work?

```
Your Question
      ↓
 [LLM ROUTER] — GPT classifies your intent
      ↓
 [HYBRID RETRIEVAL] — Two searches run together:
      ├── Keyword Search  (fast, exact match)
      └── Semantic Search (smart, meaning-based)
      ↓
 [NEO4J GRAPH] — traverses relationships:
      Brand -[:MAKES]-> Phone -[:HAS_REVIEW]-> Review
      ↓
 [AZURE OPENAI GPT] — reads context, generates answer
      ↓
    Answer ✅
```

---

## 📁 Project Structure

```
Graph_RAG/
│
├── data/
│   └── Amazon_Unlocked_Mobile.csv   ← Download from Kaggle
│
├── data_loader.py                   ← Load & clean Kaggle dataset
├── neo4j_setup.py                   ← Build Knowledge Graph in Neo4j
├── graph_query.py                   ← All Neo4j query functions
├── embeddings.py                    ← Create & search embeddings
├── rag_engine.py                    ← Complete RAG pipeline
├── main.py                          ← Run this to start chatting!
│
├── .env                             ← Your secret keys (create this)
├── .env.example                     ← Template for .env
└── requirements.txt                 ← Python packages
```

---

## 🗺️ Knowledge Graph Structure

```
(:Brand {name: "Samsung"})
        |
   [:MAKES]
        |
        ▼
(:Phone {name: "Galaxy S23", avg_rating: 4.5, price: 749})
        |
  [:HAS_REVIEW]
        |
        ▼
(:Review {text: "Amazing phone!", rating: 5, embedding: [...]})
```

**Graph Stats (with 2000 rows of Kaggle data):**
- 🟢 **92** Brand nodes
- 🔴 **914** Phone nodes  
- 🟡 **2,000** Review nodes
- 🔗 **3,306** Total nodes connected

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language (modular, no OOP) |
| **Neo4j AuraDB** | Knowledge Graph database |
| **Azure OpenAI GPT** | Answer generation |
| **Sentence Transformers** | Local embeddings (free, no API) |
| **Pandas** | Data loading & cleaning |
| **Kaggle Dataset** | Real Amazon phone reviews |

---

## 🚀 Setup Instructions

### Step 1: Clone the project
```bash
git clone https://github.com/yourusername/graph-rag-phones.git
cd graph-rag-phones
```

### Step 2: Install packages
```bash
pip install -r requirements.txt
```

### Step 3: Download Kaggle Dataset
1. Go to: https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones
2. Download `Amazon_Unlocked_Mobile.csv`
3. Put it inside a `data/` folder in the project

```
Graph_RAG/
└── data/
    └── Amazon_Unlocked_Mobile.csv  ← here
```

### Step 4: Setup Neo4j (Free!)
1. Go to https://neo4j.com/cloud/platform/aura-graph-database/
2. Click **"Create Free Instance"**
3. Save your **URI**, **username**, and **password**

### Step 5: Setup Azure OpenAI
1. Go to https://portal.azure.com
2. Create an **Azure OpenAI** resource
3. Deploy a model (e.g. `gpt-4o`)
4. Copy your **endpoint**, **key**, and **deployment name**

### Step 6: Create your `.env` file
```bash
cp .env.example .env
```

Fill in your values:
```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### Step 7: Run!
```bash
python main.py
```

That's it! First run will:
1. Load and clean the Kaggle data
2. Build the Neo4j Knowledge Graph
3. Create embeddings for all phones and reviews
4. Start the chat interface

---

## 💬 Sample Questions

```
You: Which is the best phone to buy?
You: Compare Apple vs Samsung
You: Any phone with heating issues?
You: What do customers say about OnePlus?
You: Which brand has the best rating?
You: What is the cheapest good phone?
You: Tell me about Samsung Galaxy reviews
You: Which phone has the most reviews?
```

---

## 🔍 Two Types of Search

### 1. Keyword Search (old way)
```
Query: "best phone"   → finds "best phone" ✅
Query: "achha phone"  → finds nothing      ❌
```

### 2. Semantic Search with Embeddings (new way)
```
Query: "best phone"      → works ✅
Query: "achha phone"     → works ✅ (same meaning!)
Query: "heating issues"  → finds reviews about
                           "overheats", "gets warm",
                           "battery heats up"  ✅
```

Both run together — this is called **Hybrid Search**.

---

## 🧩 How Each File Works

### `data_loader.py`
- Reads `Amazon_Unlocked_Mobile.csv`
- Cleans missing values, invalid ratings
- Limits to 2000 rows (configurable)
- Computes per-phone and per-brand statistics

### `neo4j_setup.py`
- Connects to Neo4j AuraDB
- Creates Brand, Phone, Review nodes
- Creates MAKES and HAS_REVIEW relationships
- Verifies the graph after building

### `graph_query.py`
- `get_top_phones()` — top rated phones
- `get_phones_by_brand()` — all phones by a brand
- `get_phone_reviews()` — reviews for a phone
- `get_brand_comparison()` — compare all brands
- `build_context_for_question()` — smart keyword retrieval

### `embeddings.py`
- Loads `all-MiniLM-L6-v2` model (free, local)
- Converts phone descriptions to 384-dim vectors
- Stores vectors in Neo4j as properties
- `semantic_search_phones()` — meaning-based search
- `semantic_search_reviews()` — meaning-based review search

### `rag_engine.py`
- `decide_query_type()` — LLM Router (GPT classifies intent)
- `fetch_context()` — Hybrid search (keyword + semantic)
- `generate_answer()` — GPT generates final answer
- `answer_question()` — Complete RAG pipeline

### `main.py`
- Entry point — run this file
- Auto-detects if graph needs to be built
- Shows quick stats on startup
- Runs the chat loop

---

## 📊 Graph RAG vs Normal RAG

| Feature | Normal RAG | Graph RAG |
|---|---|---|
| Storage | Vector DB | Graph DB |
| Relationships | ❌ No | ✅ Yes |
| "Brand → Phone → Review" query | ❌ Hard | ✅ Easy |
| Semantic search | ✅ Yes | ✅ Yes |
| Speed | Fast | Fast |
| Best for | Documents | Connected data |

---

## 🎯 Interview Questions This Project Covers

- What is RAG and why do we use it?
- What is a Knowledge Graph?
- What is the difference between Graph DB and SQL DB?
- What are embeddings and how do they work?
- What is cosine similarity?
- What is an LLM Router?
- What is Hybrid Search?
- What is Cypher query language?
- Why use `MERGE` instead of `CREATE` in Neo4j?
- What is `temperature` in LLM API calls?

---

## ⚠️ Common Issues

**Neo4j connection failed**
→ Check your `.env` — URI must start with `neo4j+s://`

**Embedding model downloading slowly**
→ First run downloads ~80MB — wait for it once

**Azure OpenAI error**
→ Check `AZURE_OPENAI_DEPLOYMENT_NAME` matches your actual deployment name in Azure portal

**`data/Amazon_Unlocked_Mobile.csv` not found**
→ Download from Kaggle and put in `data/` folder

---

## 📈 Future Improvements

- [ ] Add vector index in Neo4j for faster semantic search
- [ ] Add more relationship types (COMPETES_WITH, RECOMMENDED_WITH)
- [ ] Build a web UI with Streamlit
- [ ] Add price filter to queries
- [ ] Support more datasets (laptops, TVs, etc.)

---

## 👤 Author

Built by: **Anushka Pal**  
LinkedIn: https://www.linkedin.com/in/anushka-pal-a677731ba/  

---

## ⭐ If this helped you understand Graph RAG — give it a star!
