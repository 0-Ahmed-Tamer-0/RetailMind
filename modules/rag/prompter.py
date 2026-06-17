"""rag/prompter.py — Format context and call Ollama (Llama 3.1)

What this file does
────────────────────
Takes retrieved chunks + user question, builds a structured prompt,
calls the local Ollama API, and streams the response back.

Why Ollama instead of a cloud API?
────────────────────────────────────
Privacy-first design — no data leaves the machine.
Ollama runs Llama 3.1 8B as a local HTTP server on port 11434.
The API is identical to OpenAI's format, making it easy to swap
models later without changing this code.

Prompt engineering decisions
─────────────────────────────
System prompt:
  Establishes the LLM's role as a retail analyst, not a general assistant.
  Constrains it to answer ONLY from the provided context — prevents
  hallucination by making it clear the context is ground truth.
  Instructs it to say "I don't have that data" rather than guess.

Context formatting:
  Each chunk is numbered and prefixed with its source so the LLM
  can attribute answers ("According to inventory data...").
  Numbered references also enable the chatbot to show source citations.

Temperature=0.1:
  Low temperature = more deterministic, factual answers.
  We want a retail analyst, not a creative writer.
  Higher temperature would produce more varied but less reliable answers.

max_tokens=512:
  Enough for a thorough answer with recommendations.
  Prevents runaway responses that would overflow the chat UI.
"""

from __future__ import annotations

import httpx

OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

# ── Model speed configuration ────────────────────────────────────────────────
# Two modes selectable from the UI:
#   fast    → llama3.2:1b  — ~10-15s on CPU, good for factual retail queries
#   smart   → llama3.2:3b  — ~40-60s on CPU, better reasoning and synthesis
#
# The model is resolved lazily (on first call) so switching modes mid-session
# takes effect immediately on the next message.

_model_mode: str = "smart"   # default — can be changed by UI


def set_model_mode(mode: str):
    """Called by chatbot.py when user toggles fast/smart."""
    global _model_mode, _resolved_model
    _model_mode = mode
    _resolved_model = None   # force re-resolve with new preference


_resolved_model: str | None = None


def _resolve_model() -> str:
    """
    Probe Ollama for available models on first call, then cache the result.

    Lazy (not eager) because:
    - prompter.py is imported before Ollama finishes loading
    - Eager resolution at import time returns fallback "llama3.1" (wrong tag)
    - Lazy resolution runs when the first chat message is sent — Ollama is ready by then

    Tag mismatch ("llama3.1" vs "llama3.1:latest") causes HTTP 500.
    This probes /api/tags to find the exact installed tag name.
    """
    global _resolved_model
    if _resolved_model is not None:
        return _resolved_model
    try:
        r = httpx.get(OLLAMA_TAGS, timeout=3.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]

            # Choose preference order based on selected mode
            if _model_mode == "fast":
                prefixes = ["llama3.2:1b", "llama3.2", "llama3.1", "llama3", "llama"]
            else:
                prefixes = ["llama3.2:3b", "llama3.2", "llama3.1", "llama3", "llama"]

            for prefix in prefixes:
                match = next((m for m in models if m.startswith(prefix)), None)
                if match:
                    _resolved_model = match
                    print(f"[prompter] Mode={_model_mode} → model: {_resolved_model}")
                    return _resolved_model
    except Exception:
        pass
    _resolved_model = "llama3.2:3b"
    return _resolved_model

SYSTEM_PROMPT = """You are RetailMind, an AI retail analyst for a small-to-medium retail business.
You answer questions about sales forecasts, inventory levels, customer segments,
product reviews, and store foot traffic.

Rules:
- Answer ONLY using the context provided. Do not use outside knowledge.
- Be concise and business-focused. Retail owners need clear, actionable answers.
- Write in natural business language — never use [1], [2], or any numbered references.
  Integrate information naturally: "Customers mainly complain about X and Y" not "[1] shows X".
- For review/satisfaction questions: summarize the themes naturally by volume.
  Example: "Most complaints (975 reviews) are about damaged packaging,
  followed by value concerns (818 reviews)."
- For inventory questions: list products that need action with their reason.
- For segment questions: describe each segment in plain English with key numbers.
- If context does not contain enough information say exactly:
  "I don't have that data yet. Please run the relevant module first."
- Format numbers clearly: commas for thousands, £ for monetary, % for ratios.
- Never mention chunks, vectors, embeddings, RAG, or any technical AI terminology.
"""


def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into numbered context block.

    Numbering helps the LLM attribute statements to specific sources
    and helps us display citations in the Streamlit UI.
    """
    if not chunks:
        return "No relevant data found in current outputs."

    lines = ["RETAIL DATA CONTEXT:"]
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown").replace("_", " ").title()
        lines.append(f"[{i}] ({source}) {chunk['text']}")
    return "\n".join(lines)


def call_ollama(
    question:    str,
    chunks:      list[dict],
    temperature: float = 0.1,
    max_tokens:  int   = 256,
    stream:      bool  = False,
) -> str:
    """
    Call local Ollama API with retrieved context + user question.

    Parameters
    ----------
    question    : user's original question (not the embedding query)
    chunks      : retrieved chunks from retriever.retrieve()
    temperature : response randomness (0.1 = factual/deterministic)
    max_tokens  : max response length
    stream      : if True, returns generator for Streamlit st.write_stream

    Returns
    -------
    Full response string (or generator if stream=True)

    Error handling
    ──────────────
    If Ollama isn't running, raises a clear error that the pipeline.py
    fallback handler catches and routes to the keyword matcher instead.
    This is the graceful degradation point.
    """
    context = _build_context(chunks)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"{context}\n\nQuestion: {question}"},
    ]

    payload = {
        "model":   _resolve_model(),
        "messages": messages,
        "stream":  stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     2048,    # limit context window — faster on CPU
        },
    }

    for attempt in range(2):
        try:
            if stream:
                return _stream_ollama(payload)
            else:
                response = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
                response.raise_for_status()
                return response.json()["message"]["content"]

        except httpx.TimeoutException:
            if attempt == 0:
                print("[prompter] Timed out — retrying once...")
                import time; time.sleep(10)
                continue
            raise TimeoutError("Ollama timed out. Run: ollama run llama3.1 'hello' first.")
        except httpx.ConnectError:
            raise ConnectionError("Ollama not running. Start with: ollama serve")
        except httpx.TimeoutException:
            raise TimeoutError(
                "Ollama timed out (>120s). The model may still be loading.\n"
                "Try again in 30 seconds."
            )


def _stream_ollama(payload: dict):
    """
    Generator that yields text tokens as they arrive from Ollama.
    Compatible with Streamlit's st.write_stream().
    """
    with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as response:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                import json
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except Exception:
                continue


def ollama_available(timeout: float = 2.0) -> bool:
    """
    Check if Ollama is running without raising an exception.
    Used by pipeline.py to decide RAG vs keyword fallback.

    timeout=2.0: fast enough for UI responsiveness,
    long enough for a loaded server to respond.
    """
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
