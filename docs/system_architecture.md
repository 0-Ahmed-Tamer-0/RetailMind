# System Architecture – AI Retail Decision Support System

## 1. Overview
The AI Retail Decision Support System is designed as a modular, data-driven
architecture that integrates multiple AI components to support business owners
in making informed decisions.

The system focuses on analysis, explanation, and recommendation rather than
full automation.

---

## 2. High-Level Architecture
The system consists of four main layers:

1. Data Layer
2. AI Processing Layer
3. Integration & Storage Layer
4. Decision Interface Layer

---

## 3. Data Layer
This layer is responsible for collecting raw data from different sources.

### Data Sources:
- In-store camera data (foot traffic)
- POS transaction data
- Customer reviews (text data)
- Customer purchase history

Each data source is processed independently and stored in structured formats
(CSV files) following predefined schemas.

---

## 4. AI Processing Layer
This layer contains independent AI modules, each responsible for a specific task.

### 4.1 Computer Vision Module
- Input: Camera footage or simulated foot traffic data
- Output: Number of customers per zone and time window
- Purpose: Analyze customer flow and store congestion

### 4.2 Sales Forecasting Module
- Input: Historical POS data
- Output: Predicted future sales
- Purpose: Support inventory and staffing decisions

### 4.3 Customer Segmentation Module
- Input: Customer transaction history
- Output: Customer clusters
- Purpose: Identify customer behavior patterns

### 4.4 Review Analysis Module
- Input: Customer reviews
- Output: Sentiment and key complaint topics
- Purpose: Measure customer satisfaction

Each module produces structured outputs that follow a predefined schema.

---

## 5. Integration & Storage Layer
All AI module outputs are saved as CSV files in a shared output directory.
These files act as the communication medium between modules.

A business KPI aggregation process uses these outputs to compute
high-level performance indicators.

This layer ensures loose coupling between AI modules.

---

## 6. Decision Interface Layer
This layer is responsible for presenting insights to business owners.

### 6.1 Dashboard
- Displays KPIs and trends
- Provides visual summaries of system outputs

### 6.2 AI Chatbot
- **Dual-Engine Architecture**: Operates in RAG (Retrieval-Augmented Generation) mode when local inference is available, falling back to a structured rule-based keyword matcher on constrained hardware.
- **Local LLM**: Uses Llama 3.1 8B (or Llama 3.2) hosted locally via Ollama to generate natural language answers and explain business metrics.
- **RAG Pipeline**: Pre-formats CSV tabular outputs into human-readable text chunks, embeds them locally using `all-MiniLM-L6-v2`, and stores them in a local FAISS vector database.
- **Intent Routing**: Automatically detects the user's intent to apply source-specific context filters (e.g. inventory queries are restricted to inventory chunks) to prevent token waste and noise.
- **Fallback Engine**: Implements regular expression entity matching and rule-based template generation when the Ollama server is unreachable.
- **Visual Payloads**: Generates Plotly graphs and zone traffic heatmaps (overlay images) in response to visual query intents.


---

## 7. Design Principles
- Modularity: Each AI module operates independently
- Scalability: New modules can be added without redesign
- Explainability: Focus on insights, not black-box decisions
- Business-Centric Design: Outputs are aligned with business KPIs

---

## 8. System Limitations
- The system depends on data quality and availability
- Real-time processing is not required in the current scope
- Recommendations are advisory, not autonomous actions
