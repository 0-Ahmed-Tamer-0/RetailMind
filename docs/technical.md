# RetailMind — Technical Upgrade Log

Every decision made during the v1 → v2 rebuild, with the engineering and
scientific justification behind each one. Written to be referenced during
academic discussion, technical interviews, or code review.

---

## 1. Sales Forecasting

### 1.1 Daily → Weekly Aggregation

**v1:** Aggregated transactions by calendar date. Prophet received ~730 data
points per product over two years.

**v2:** Aggregated by ISO week (Monday anchor, `freq="W-MON"`). Prophet
receives ~104 data points per product.

**Why:**
Daily retail transaction data has high day-to-day variance caused by order
timing, not real demand changes — a product can show £5,000 on Monday and
£0 on Tuesday simply because orders batch on weekdays. This noise causes
Prophet to either overfit (treating every spike as a meaningful pattern) or
underfit (averaging trends to compensate). Weekly aggregation removes
day-of-week noise while preserving monthly cycles, seasonal peaks, and
structural trend changes. Result: the forecast line tracks the Christmas
gift-ordering spike (October–November in UCI data) instead of producing a
flat underestimate.

---

### 1.2 Tuned Prophet Hyperparameters

**v1:** Default Prophet configuration (`changepoint_prior_scale=0.05`,
additive seasonality).

**v2:** Three targeted changes:

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `changepoint_prior_scale` | 0.05 | 0.15 | Default is too rigid for retail with genuine structural breaks |
| `seasonality_mode` | `additive` | `multiplicative` | Retail Christmas spike is proportional to revenue level — a £5K/week product has a bigger absolute spike than a £500/week product |
| `seasonality_prior_scale` | 10 (default) | 10.0 (explicit) | Ensures strong seasonal component is learned from 2 years of data |
| `weekly_seasonality` | True | False | Weekly-aggregated data has no intra-week pattern left to model |

**Why multiplicative matters:**
Additive seasonality assumes Christmas adds a fixed £X to every product.
Multiplicative assumes it multiplies revenue by a factor — which is
empirically true in retail. Using additive on a high-revenue product
systematically underestimates the peak.

---

### 1.3 Confidence Intervals in Output

**v1:** Output contained only `predicted_sales`.

**v2:** Output contains `predicted_sales`, `predicted_low`, `predicted_high`
(80% confidence interval from Prophet).

**Why:**
Two downstream consumers use these columns:
1. **Inventory module** — derives dynamic safety stock from interval width
   (wider interval = more uncertainty = larger buffer)
2. **Dashboard/chat** — enables the system to say "between X and Y units"
   instead of a single overconfident number

---

## 2. Customer Segmentation

### 2.1 Synthetic Features → Real RFM Features

**v1:** Segmentation used fabricated columns: `age`, `satisfaction_score`,
`promotion_usage`. These do not exist in any real transaction dataset.

**v2:** Features computed directly from UCI transaction history:

| Feature | Definition | Computed from |
|---|---|---|
| **Recency** | Days since last purchase | `max(InvoiceDate)` per customer |
| **Frequency** | Number of distinct invoices | `nunique(Invoice)` per customer |
| **Monetary** | Total spend | `sum(Quantity × Price)` per customer |

**Why RFM:**
RFM is the industry standard for customer segmentation — used by Amazon,
Shopify Analytics, and every major CRM platform. Every number is traceable
to a real invoice row in the raw data. This makes the segmentation
academically defensible and practically meaningful.

---

### 2.2 Automatic Elbow Detection

**v1:** Hardcoded `k=4` clusters.

**v2:** `best_k()` function computes inertia for k=2..8, detects the
knee point using second-order finite differences, and selects k
automatically. Falls back to k=4 if detection is ambiguous.

**Why:**
Hardcoding k is arbitrary. The knee-point method finds the k where adding
more clusters yields diminishing improvement — a principled, data-driven
choice.

---

### 2.3 Rank-Based Segment Naming

**v1:** Fixed threshold-based naming (e.g., `recency < 30 → Champion`).

**v2:** Clusters ranked relative to each other across all three RFM
dimensions. Champions are the cluster with the lowest recency AND highest
frequency combined — not any cluster below an arbitrary day threshold.

**Why:**
Fixed thresholds break when applied to different datasets. Rank-based
naming adapts to the actual distribution in your data.

---

## 3. Reviews & Sentiment Analysis

### 3.1 Single Signal → Three-Signal Majority Vote

**v1:** Sentiment label derived from star rating alone (`Score >= 3 →
positive`).

**v2:** Three independent signals, majority vote (2/3 required):

| Signal | Source | Technology |
|---|---|---|
| Score-based | Star rating | Rule: ≥4=pos, ≤2=neg, 3=neutral |
| VADER | Review text | Lexicon-based NLP compound score |
| TextBlob | Review text | Pattern-based polarity analysis |

**Why three signals:**
A 1-star review can contain positive text ("surprisingly good for the
price"). A 5-star review can contain a buried complaint. Single-signal
methods fail both cases. Two independent NLP models agreeing with the
star rating gives much higher confidence. Conflicted reviews (signals
disagree) are labelled neutral — the honest answer when evidence is
mixed.

---

### 3.2 3-Class Labelling

**v1:** Binary: positive / negative.

**v2:** Three classes: positive / negative / neutral.

**Why:**
3-star reviews are genuinely mixed. Calling them positive inflates
satisfaction metrics and hides real customer dissatisfaction. The neutral
class captures ambiguous sentiment rather than forcing a false binary.

---

### 3.3 LDA Topic Modeling

**v1:** Keyword frequency counts. Output: a list of words sorted by
occurrence.

**v2:** Latent Dirichlet Allocation (LDA) on each sentiment class
separately. Output: named business themes with review counts.

**Why:**
Keywords answer "what words appear?" Topics answer "what are customers
actually talking about?" LDA finds co-occurrence patterns — words like
`stale, expired, smell, rotten` appearing together in the same reviews
form a topic: "Freshness & Expiry Issues." A retail owner can act on
that. They cannot act on "stale: 142 occurrences."

---

### 3.4 Corpus-Based Stopword Induction

**v1:** No domain stopword handling.

**v2:** Offline script (`induce_stopwords.py`) computes two statistics
for every word in the full 568K review corpus:

- **Document frequency (DF):** what % of reviews contain this word
- **Discriminative power:** |freq_in_positive − freq_in_negative|

A word is a stopword if `DF > 15%` (too common) OR
`discriminative_power < 0.5%` (appears equally in both classes).

**Why:**
Manual stopword lists are brittle and arbitrary. Corpus-induced stopwords
reflect the actual statistical properties of your data. The thresholds
(15%, 0.5%) are documented and explainable — not guesses.

---

### 3.5 POS Filtering for LDA Input

**v2 addition:** Before LDA training, tokens are filtered to nouns (NN,
NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS) using NLTK's
`averaged_perceptron_tagger`.

**Why:**
Topics are defined by things (nouns) and their properties (adjectives).
Verbs and adverbs describe actions — they cannot name a topic. Without
POS filtering, LDA anchors topics around high-frequency verbs like
`use, find, try` that appear in every review and mean nothing about
product category. POS filtering is a linguistically principled approach,
not another heuristic word list.

---

### 3.6 Semantic Topic Naming (MiniLM Embeddings)

**v1:** No topic naming (no LDA in v1).

**v2 initial attempt:** Rule-based `if/else` matching top words to theme
strings — rejected as indefensible.

**v2 final:** The top-word phrase for each LDA topic is embedded with
`all-MiniLM-L6-v2`. A candidate vocabulary of 20 retail-domain theme
names is also embedded. Cosine similarity selects the closest theme.

**Why:**
The embedding model understands that "rancid smell expired old" is
semantically close to "Freshness & Expiry Issues" even though none of
those words appear in the theme string. It generalises from meaning,
not string matching. The candidate vocabulary is domain knowledge —
changing it requires editing a list, not rewriting logic.

---

## 4. Inventory Management

### 4.1 Fixed Safety Stock → Dynamic from Forecast Uncertainty

**v1:** `safety_stock = predicted_demand × 0.20` for every product.

**v2:**
```
uncertainty_ratio = mean(predicted_high − predicted_low) / mean(predicted_sales)
safety_stock_pct  = clamp(uncertainty_ratio × abc_multiplier, 0.10, 0.50)
```

**Why:**
A product with a tight, confident forecast (narrow Prophet interval)
needs less buffer than a volatile product with a wide interval. Using
the same 20% for both either wastes capital (over-ordering confident
products) or under-protects volatile ones. The confidence interval from
Prophet becomes a direct input to inventory decisions — two modules
talking to each other through a shared signal.

---

### 4.2 ABC Classification

**v1:** Same reorder rules for all products.

**v2:** Products classified A/B/C by cumulative revenue contribution:
- A: top 50% of revenue → earlier reorder trigger, higher safety buffer
- B: next 30% → standard rules
- C: bottom 50% → later trigger, lower discount threshold

**Why cumulative share, not top-N:**
"Top 3 products" means something very different if one product has 90%
of revenue vs 25%. Cumulative share adapts to the actual revenue
distribution. This is the standard Pareto-based ABC method used in
every retail ERP system.

---

## 5. RAG Pipeline (New in v2)

### 5.1 Architecture

**v1:** Keyword router. User intent detected by fuzzy string matching,
hardcoded function called, hardcoded response returned.

**v2:** Full RAG (Retrieval-Augmented Generation) pipeline:

```
User question
      ↓
Embed with all-MiniLM-L6-v2 (384-dim vector)
      ↓
FAISS IndexFlatIP search over 60+ pre-built chunks
      ↓
Top-3 most relevant chunks retrieved
      ↓
Prompt Llama 3.2 (Ollama, local) with retrieved context
      ↓
Natural language answer with source citations
```

**Why RAG over fine-tuning:**
Fine-tuning requires labelled Q&A pairs we don't have. RAG grounds the
LLM in real output data — it cannot hallucinate numbers that aren't in
the retrieved chunks. Every answer is traceable to a source CSV.

---

### 5.2 Human-Readable Chunk Design

**v1:** N/A.

**v2:** Each CSV row is converted to a natural language sentence before
embedding. Example:

```
Raw CSV row:
product_id=85123A, decision=REORDER, current_stock=980, ...

Chunk text:
"Inventory status for product 85123A (ABC class A): current stock 980 units,
7-day forecast demand 1134 units. Decision: REORDER. A-class product ·
forecast 1134 units · safety buffer 28% · required 1458 > stock 980."
```

**Why:**
The LLM reads chunks as context. Raw CSV rows produce worse answers
than pre-formatted natural language. The LLM spends less of its context
window parsing structure and more on reasoning about content.

---

### 5.3 Aggregated Review Topic Chunks

Early v2 design: one chunk per LDA topic (10 chunks total).

Final v2 design: two chunks — all negative topics aggregated, all
positive topics aggregated.

**Why:**
A satisfaction query needs the LLM to see the complete picture — all
complaint themes AND all praise themes — to synthesise a balanced answer.
With individual chunks, retrieval might return only the 1-2 topics
closest to the query vector, producing a partial answer. Aggregated
chunks give the full picture in one retrieval hit.

---

### 5.4 Intent-Based Source Filtering

Before FAISS search, query intent is detected and a `source_filter`
restricts retrieval to the most relevant CSV sources.

Special cases bypass semantic retrieval entirely:
- **Satisfaction queries** → force-load all review_topics chunks
  (semantic search would return segments, not reviews)
- **Top-N forecast queries** → dump all sales chunks (not semantic)
- **Visual queries** → bypassed entirely, chart built directly

**Why:**
Semantic search finds what is most similar to the query vector —
not always what is most useful. "Are customers satisfied?" has high
cosine similarity to segment chunks (loyal, champions) but the correct
answer requires review data. Explicit routing for known failure modes
produces better answers than relying on embeddings alone.

---

### 5.5 Adaptive Inference with Graceful Degradation

**v2:** Runtime detection of Ollama availability. Two modes:

| Mode | Trigger | Response |
|---|---|---|
| RAG + Llama 3.2 | Ollama running + user selected Ollama | Intelligent, ~40s |
| Keyword matcher | Ollama offline OR user selected Rule-based | Instant, always works |

User-selectable toggle in sidebar. Ollama button disabled automatically
when Ollama is not running (prevents confusing errors).

**Why:**
A system that only works on high-spec hardware is not a usable product.
The fallback ensures the system remains functional on any machine. It
also demonstrates understanding of production resilience — a genuine
software engineering consideration, not just an academic prototype.

---

## 6. UI / UX

### 6.1 Branded Layout

**v1:** Default Streamlit layout. `st.title()` header, no visual hierarchy.

**v2:** Custom CSS throughout:
- Branded header bar with system name and version tag
- KPI strip auto-computed from CSV outputs (5 live metrics)
- Card-based layout with accent colors per metric type
- DM Sans font via Google Fonts import

---

### 6.2 Live KPI Strip

**v2 new:** 5 KPI cards at the top of every page:
- Top product forecast (units)
- Items to reorder (count)
- Average review rating
- Customer segments (count)
- Peak foot traffic (people)

All computed at runtime from CSV outputs — not hardcoded. Updates
automatically when outputs are regenerated.

---

### 6.3 Streaming Responses

**v1:** Full response generated server-side, displayed all at once after
40-90 second wait.

**v2:** Token streaming via Ollama's streaming API. First tokens appear
within 2-3 seconds. Total generation time unchanged but perceived latency
drops dramatically.

**Why:**
All production LLM interfaces (ChatGPT, Claude, Gemini) stream for this
reason. A blank screen for 60 seconds is psychologically much worse than
tokens appearing progressively after 2 seconds.

---

### 6.4 Inline Visual Responses

**v2 new:** Visual intents (heatmap, plot, chart) bypass the RAG pipeline
entirely and return Plotly figures directly in chat:

- `plot product X` → actual vs forecast chart with confidence band
- `show heatmap` → overlay PNG images from CV module
- `busiest zones chart` → bar chart by zone
- `rating trend chart` → dual-axis sentiment trend

**Why separate from RAG:**
RAG returns text. Charts require a Plotly figure object. Mixing these
in the same pipeline would require the LLM to generate Plotly code —
unreliable and slow. Visual detection before the RAG call keeps the
two concerns cleanly separated.

---

## 7. Dataset Upgrades

| Module | v1 Data | v2 Data |
|---|---|---|
| Sales / Inventory / Segmentation | Synthetic / small sample | UCI Online Retail II — 1M+ real UK transactions 2009-2011 |
| Reviews | Synthetic ratings | Amazon Fine Food Reviews — 568K real reviews |
| CV Foot Traffic | Existing zone images | Pretrained YOLOv8n (COCO person class) |

**Why UCI Online Retail II for three modules:**
One real dataset feeds three modules (sales, inventory, segmentation) —
they share the same cleaning logic (`clean_uci()` in `forecast_model.py`)
ensuring consistency. A single data source is more defensible than three
separate synthetic datasets with invented columns.

---

## 8. Privacy-First Architecture

Every component runs locally:

| Component | Technology | Internet required? |
|---|---|---|
| LLM inference | Ollama + Llama 3.2 | ❌ After initial pull |
| Embeddings | sentence-transformers MiniLM | ❌ After initial download |
| Vector store | FAISS (disk-based) | ❌ Never |
| UI | Streamlit (localhost:8501) | ❌ Never |
| Data | Local CSV files | ❌ Never |

**Why this matters:**
Retail transaction data and customer reviews are sensitive business data.
A cloud-dependent system means customer data leaves the premises on every
query. A fully local system means the privacy guarantee is architectural,
not policy-based — it is physically impossible for data to be transmitted
externally during normal operation.