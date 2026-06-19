# 🏪 RetailMind

**An offline, privacy-first AI decision support system for small and medium-sized retailers.**

RetailMind integrates computer vision, demand forecasting, inventory management, sentiment analysis, and customer segmentation into a single conversational interface — running entirely on your own machine. No cloud. No subscription. No data ever leaves your laptop.

Graduation project — Faculty of Computers and Artificial Intelligence, Benha University (2025–2026).

---

## Why RetailMind

E-commerce platforms know in real-time which products are about to sell out and which customers are about to churn. Physical store owners are still guessing — operating on intuition with no way to turn their own data (transactions, reviews, CCTV footage) into answers.

Enterprise retail analytics platforms solve this but cost $15,000+ per year and require IT infrastructure. RetailMind provides the same core capabilities for **zero recurring cost**, on **any laptop**, with **architectural privacy guarantees** rather than policy promises.

---

## What it does

| Module | Technique | Output |
|---|---|---|
| 🚶 **Foot Traffic** | YOLOv8n person detection + Zone 41 spatial sub-splitting | Zone occupancy heatmaps, busiest-zone rankings |
| 📈 **Sales Forecasting** | Facebook Prophet, weekly aggregation, multiplicative seasonality | 30-day demand forecasts with 80% confidence intervals |
| 📦 **Inventory Management** | ABC classification + dynamic safety stock from forecast uncertainty | Reorder recommendations, discount suggestions |
| 💬 **Review Analysis** | 3-signal sentiment (VADER + TextBlob + rating) + LDA topic modeling | Named complaint/praise themes, sentiment trends |
| 👥 **Customer Segmentation** | RFM features + K-Means with automatic elbow detection | Champions / Loyal / At-Risk customer groups |
| 🤖 **RAG Chatbot** | FAISS + MiniLM embeddings + Llama 3.2 (Ollama) | Natural language answers grounded in your actual data |

All five modules communicate through a shared CSV output layer — fully decoupled, independently testable, and trivially extensible.

---

## Architecture

```
Raw Data (local files)
   ├─ UCI Online Retail II ──┬─→ Sales Forecasting   ──┐
   │                          ├─→ Inventory Mgmt       │
   │                          └─→ Customer Segments    │
   ├─ Amazon Fine Food Reviews → Review Analysis       ├─→ data/outputs/*.csv
   └─ CCTV Zone Images ────────→ Foot Traffic CV       │
                                                         ↓
                                              RAG Indexer (MiniLM + FAISS)
                                                         ↓
                                    Streamlit Chat ←─ Llama 3.2 (Ollama)
                                         ↑
                              Keyword Fallback (if Ollama offline)
```

**Privacy guarantee is architectural, not policy-based:** Streamlit binds to `localhost:8501`, Ollama binds to `localhost:11434`. No HTTP request to any external server occurs during normal operation.

---

## Quick Start

### 1. Install Ollama and pull a model
```bash
# Download from https://ollama.ai, then:
ollama pull llama3.2:3b
```

### 2. Clone and install dependencies
```bash
git clone https://github.com/your-org/retailmind.git
cd retailmind
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon stopwords averaged_perceptron_tagger
```

### 3. Launch the app
```bash
streamlit run modules/chatbot/chatbot.py
```

### 4. Configure your data
Open the **Setup** tab and enter the paths to your datasets:
- UCI Online Retail II `.xlsx` (or your own POS export in the same schema)
- Review CSV (star rating + text columns)
- CCTV zone image folders
- *(Optional)* existing stock CSV — auto-generated from sales history if omitted

Click **Run All Modules**, then switch to the **Chat** tab.

---

## Example queries

```
what should I reorder?
are my customers satisfied with my store?
which zone is busiest?
plot product 85123A
show heatmap
describe my customer segments
what do customers love?
```

Every AI-mode answer shows its source CSV and a mode badge (🤖 Llama 3.2 vs 🔤 Rule-based), so you always know what generated the response.

---

## Project structure

```
modules/
  chatbot/              # Streamlit UI + rule-based fallback engine
  rag/                  # FAISS indexer, retriever, Ollama prompter, pipeline orchestrator
  sales_forecasting/    # Prophet model + output generator
  inventory_management/ # ABC classification + dynamic safety stock logic
  customer_segmentation/# RFM + K-Means + elbow detection
  reviews_analysis/     # VADER/TextBlob/LDA pipeline + stopword induction
  cv_foot_traffic/      # YOLOv8n detection + zone overlay generation
data/
  raw/        # Your input datasets (not tracked in git)
  outputs/    # Generated CSVs — the shared communication layer between modules
  rag_index/  # FAISS index + chunk metadata
```

---

## Datasets used for validation

| Dataset | Source | Size |
|---|---|---|
| UCI Online Retail II | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) | 1M+ UK transactions (2009–2011) |
| Amazon Fine Food Reviews | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) | 568K reviews |

Both are real public datasets — no synthetic placeholders.

---

## Tech stack

`Python` · `Streamlit` · `Ollama` + `Llama 3.2` · `FAISS` · `Sentence-Transformers` · `Facebook Prophet` · `YOLOv8n (Ultralytics)` · `Gensim LDA` · `scikit-learn` · `VADER` · `TextBlob` · `Plotly` · `OpenCV`

---

## Known limitations

- CPU-only inference: 40–60s response time with Llama 3.2 3B (mitigated by streaming, not eliminated)
- Single-session chatbot memory — no multi-turn conversation history yet
- Static CCTV snapshots only, not live video streams
- YOLOv8n uses pretrained COCO weights — not fine-tuned on retail-specific imagery

---

