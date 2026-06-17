"""chatbot.py  –  AI Retail Decision Support System
Streamlit UI (redesigned)

Run from repo root:
    streamlit run modules/chatbot/chatbot.py

Changes vs original:
- Professional branded header & sidebar with system health indicators
- KPI summary strip (live from CSV outputs) above the chat
- Cleaner chat bubbles with source badges
- Dashboard tab with Plotly charts instead of raw JSON
- Setup tab redesigned with step indicators
- Custom CSS theme (dark-friendly, card-based layout)
- Removed tkinter file-dialog (not available on servers); path text inputs kept
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from modules.chatbot.response_logic import (
    ChatContext,
    load_outputs,
    outputs_status,
)
from modules.rag.pipeline import get_response

# ─────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RetailMind – AI Decision Support",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── branded top bar ── */
    .brand-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px 0 8px;
        border-bottom: 1.5px solid rgba(255,255,255,0.08);
        margin-bottom: 20px;
    }
    .brand-logo {
        font-size: 28px;
        line-height: 1;
    }
    .brand-name {
        font-size: 22px;
        font-weight: 600;
        letter-spacing: -0.5px;
        color: var(--text-color);
    }
    .brand-tag {
        font-size: 12px;
        color: #888;
        margin-left: auto;
        background: rgba(255,255,255,0.06);
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* ── KPI strip ── */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--kpi-accent, #4f8ef7);
        border-radius: 2px 2px 0 0;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 500;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 600;
        color: var(--text-color);
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 11px;
        color: #666;
        margin-top: 3px;
    }
    .kpi-ok   { --kpi-accent: #22c55e; }
    .kpi-warn { --kpi-accent: #f59e0b; }
    .kpi-info { --kpi-accent: #4f8ef7; }
    .kpi-purple { --kpi-accent: #a78bfa; }
    .kpi-coral  { --kpi-accent: #f87171; }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.15);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-section {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #555;
        margin: 18px 0 8px;
    }
    .status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-ok   { background: #22c55e; }
    .dot-miss { background: #ef4444; }

    /* ── chat messages ── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 6px !important;
    }

    /* ── quick action pills ── */
    .quick-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 10px 0 16px;
    }

    /* ── tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-size: 13px !important;
    }

    /* ── step badge (setup tab) ── */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px; height: 24px;
        border-radius: 50%;
        background: #4f8ef7;
        color: #fff;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .step-title {
        font-size: 15px;
        font-weight: 600;
        margin: 20px 0 8px;
        display: flex;
        align-items: center;
    }

    /* ── dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── sidebar status dots ── */
    [data-testid="stSidebar"] .stMarkdown p { margin: 2px 0; }

    /* ── module run button row ── */
    .run-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_cached_outputs() -> dict:
    return load_outputs()


def _refresh_data_cache() -> dict:
    st.cache_data.clear()
    return _load_cached_outputs()


def _render_payload(payload: dict, msg_idx: int = 0):
    if payload.get("text"):
        st.markdown(payload["text"])
    table = payload.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.dataframe(table, use_container_width=True)
    fig = payload.get("fig")
    if fig is not None:
        st.plotly_chart(
            fig, use_container_width=True,
            key=f"chart_{msg_idx}_{id(fig)}",
        )
    # Primary image
    img = payload.get("image_path")
    if img and isinstance(img, str):
        st.image(img, use_container_width=True)
    # Extra images — fix: iterate properly, check each is a valid string path
    extra_imgs = payload.get("extra_images")
    if isinstance(extra_imgs, list) and extra_imgs:
        valid = [p for p in extra_imgs if isinstance(p, str) and p]
        if valid:
            st.markdown("**All zone overlays:**")
            cols = st.columns(min(len(valid), 3))
            for idx_img, p in enumerate(valid[:6]):
                cols[idx_img % 3].image(p, use_container_width=True)


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None
if "ctx" not in st.session_state:
    st.session_state.ctx = ChatContext()
if "data" not in st.session_state:
    st.session_state.data = _load_cached_outputs()


# ─────────────────────────────────────────────
# KPI helpers  (derived from loaded CSV outputs)
# ─────────────────────────────────────────────
def _compute_kpis(data: dict) -> dict:
    kpis: dict = {}

    # Sales: top product forecast sum
    sales = data.get("sales")
    if isinstance(sales, pd.DataFrame) and not sales.empty:
        # Column is predicted_sales (from forecast_model.py output)
        pred_col = next((c for c in ["predicted_sales","yhat","sales"] if c in sales.columns), None)
        if pred_col and "product_id" in sales.columns:
            top = sales.groupby("product_id")[pred_col].sum().nlargest(1)
            kpis["top_sales"] = f"{int(top.values[0]):,}" if len(top) else "–"
            kpis["top_sales_pid"] = str(top.index[0]) if len(top) else ""
        else:
            kpis["top_sales"] = "–"; kpis["top_sales_pid"] = ""
    else:
        kpis["top_sales"] = "–"; kpis["top_sales_pid"] = ""

    # Inventory: reorder count
    inv = data.get("inventory")
    if isinstance(inv, pd.DataFrame) and "decision" in inv.columns:
        kpis["reorder_count"] = int((inv["decision"] == "REORDER").sum())
    else:
        kpis["reorder_count"] = "–"

    # Sentiment: avg rating
    ri = data.get("review_insights")
    if isinstance(ri, pd.DataFrame) and "rating" in ri.columns:
        kpis["avg_rating"] = f"{ri['rating'].mean():.1f}"
    elif isinstance(ri, pd.DataFrame) and "avg_rating" in ri.columns:
        kpis["avg_rating"] = f"{ri['avg_rating'].mean():.1f}"
    else:
        kpis["avg_rating"] = "–"

    # Segments count
    seg = data.get("segments")
    if isinstance(seg, pd.DataFrame) and "segment_name" in seg.columns:
        kpis["segment_count"] = seg["segment_name"].nunique()
    else:
        kpis["segment_count"] = "–"

    # Foot traffic: busiest zone peak
    cv = data.get("cv")
    if isinstance(cv, pd.DataFrame) and "people_count" in cv.columns and "zone_id" in cv.columns:
        peak = cv.groupby("zone_id")["people_count"].max()
        kpis["peak_traffic"] = int(peak.max()) if len(peak) else "–"
        kpis["peak_zone"] = str(peak.idxmax()) if len(peak) else ""
    else:
        kpis["peak_traffic"] = "–"; kpis["peak_zone"] = ""

    return kpis


def _render_kpi_strip(data: dict):
    kpis = _compute_kpis(data)
    st.markdown(
        f"""
        <div class="kpi-strip">
          <div class="kpi-card kpi-info">
            <div class="kpi-label">Top product forecast</div>
            <div class="kpi-value">{kpis['top_sales']}</div>
            <div class="kpi-sub">units · {kpis['top_sales_pid']}</div>
          </div>
          <div class="kpi-card kpi-warn">
            <div class="kpi-label">Items to reorder</div>
            <div class="kpi-value">{kpis['reorder_count']}</div>
            <div class="kpi-sub">flagged by inventory model</div>
          </div>
          <div class="kpi-card kpi-ok">
            <div class="kpi-label">Avg review rating</div>
            <div class="kpi-value">{kpis['avg_rating']}</div>
            <div class="kpi-sub">from customer reviews</div>
          </div>
          <div class="kpi-card kpi-purple">
            <div class="kpi-label">Customer segments</div>
            <div class="kpi-value">{kpis['segment_count']}</div>
            <div class="kpi-sub">identified by clustering</div>
          </div>
          <div class="kpi-card kpi-coral">
            <div class="kpi-label">Peak traffic</div>
            <div class="kpi-value">{kpis['peak_traffic']}</div>
            <div class="kpi-sub">persons · zone {kpis['peak_zone']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 4px;">
            <div style="font-size:20px;font-weight:700;letter-spacing:-0.5px;">🏪 RetailMind</div>
            <div style="font-size:11px;color:#666;margin-top:2px;">AI Decision Support · v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Quick questions</div>', unsafe_allow_html=True)

    quick_actions = [
        ("📈", "Top-10 forecast",      "what are the top 10 products by forecast?"),
        ("📦", "Reorder list",          "what should I reorder?"),
        ("🏷️", "Discount picks",        "which products need a discount?"),
        ("💬", "Top complaints",        "what are customers complaining about?"),
        ("😍", "Top praises",           "what do customers love?"),
        ("😊", "Customer satisfaction", "are my customers satisfied with my store?"),
        ("👥", "Segments",              "describe my customer segments"),
        ("🚶", "Busiest zones",         "busiest zones chart"),
        ("🔥", "Zone heatmap",          "show heatmap"),
        ("⭐", "Rating trend",          "rating trend chart"),
    ]

    for icon, label, msg in quick_actions:
        if st.button(f"{icon} {label}", use_container_width=True):
            st.session_state.pending_message = msg

    st.divider()
    st.markdown('<div class="sidebar-section">Response mode</div>', unsafe_allow_html=True)
    _mode = st.session_state.get("response_mode", "ollama")

    # Ollama availability indicator
    try:
        from modules.rag.prompter import ollama_available
        _ollama_up = ollama_available()
    except Exception:
        _ollama_up = False
    st.markdown(
        f'<div style="font-size:11px;color:{"#22c55e" if _ollama_up else "#ef4444"};margin-bottom:8px;">'
        f'{"🟢 Ollama running" if _ollama_up else "🔴 Ollama offline"}</div>',
        unsafe_allow_html=True,
    )

    _c1, _c2 = st.columns(2)
    with _c1:
        if st.button(
            "⚡ Rule-based", use_container_width=True,
            type="primary" if _mode == "keyword" else "secondary",
            help="Instant · always works · no Ollama needed",
        ):
            st.session_state.response_mode = "keyword"
            st.rerun()
    with _c2:
        if st.button(
            "🤖 Ollama", use_container_width=True,
            type="primary" if _mode == "ollama" else "secondary",
            help="AI-powered · ~40s · requires Ollama running",
            disabled=not _ollama_up,
        ):
            st.session_state.response_mode = "ollama"
            st.rerun()

    _desc = {
        "keyword": "⚡ Instant · rule-based responses",
        "ollama":  "🤖 AI · llama3.2 · ~40s per response",
    }
    st.caption(_desc.get(_mode, ""))
    st.divider()
    st.caption("Plot a product: *plot product 85123A*")


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-bar">
        <span class="brand-logo">🏪</span>
        <span class="brand-name">RetailMind</span>
        <span class="brand-tag">AI Decision Support System</span>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_kpi_strip(st.session_state.data)

# Persist active tab so st.rerun() after chat submission stays on Chat tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

_tab_labels = ["💬 Chat", "📊 Dashboard", "⚙️ Setup", "ℹ️ About"]
tab_chat, tab_dashboard, tab_setup, tab_about = st.tabs(_tab_labels)


# ────────────────────────────────
# TAB 1 · Chat
# ────────────────────────────────
# ── Process any pending message BEFORE rendering tabs ──────────────────────
# This must run outside the tab context so sidebar quick-action buttons
# and chat input both route through the same processing path.
# The result is stored in session_state.messages and rendered in the tab below.
if st.session_state.get("pending_message"):
    msg = str(st.session_state.pending_message)
    st.session_state.pending_message = None
    st.session_state.messages.append({"role": "user", "content": msg})

    # Check if this is a visual request — visuals don't stream
    from modules.rag.pipeline import (
        _PLOT_KEYWORDS, _RATING_KEYWORDS, _HEATMAP_KEYWORDS,
    )
    q_lower_check = msg.lower()
    is_visual = (
        any(k in q_lower_check for k in _HEATMAP_KEYWORDS) or
        any(k in q_lower_check for k in _RATING_KEYWORDS) or
        any(k in q_lower_check for k in _PLOT_KEYWORDS) or
        any(k in q_lower_check for k in {
            "busiest zone", "busiest zones", "which zone",
            "zone traffic", "zone chart",
        })
    )

    if is_visual:
        # Visual responses: no streaming, use spinner
        with st.spinner("Building chart…"):
            payload, updated_ctx = get_response(
                msg, st.session_state.data, st.session_state.ctx
            )
        st.session_state.ctx = updated_ctx
        st.session_state.messages.append({"role": "assistant", "payload": payload})
        st.rerun()
    else:
        # Text responses: stream tokens live
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(msg)

        with st.chat_message("assistant", avatar="🏪"):
            stream_placeholder = st.empty()
            full_text = ""

            try:
                from modules.rag.pipeline import (
                    _detect_sources, _HEATMAP_KEYWORDS,
                )
                from modules.rag.retriever import retrieve, retrieve_by_source
                from modules.rag.prompter import (
                    call_ollama, ollama_available, _build_context,
                )

                _mode_check = st.session_state.get("response_mode", "ollama")
                if _mode_check != "keyword" and ollama_available():
                    # Retrieve chunks
                    satisfaction_words = {
                        "satisfied", "satisfaction", "happy with",
                        "customers think", "customers say", "are customers",
                        "good reviews", "bad reviews",
                    }
                    if any(w in q_lower_check for w in satisfaction_words):
                        chunks = retrieve_by_source("review_topics", top_n=10)
                        chunks += retrieve(msg, top_k=2,
                                          source_filter=["review_insights"])
                    elif any(w in q_lower_check for w in {
                        "top 10", "top products", "all products",
                        "highest forecast", "most demand",
                    }):
                        chunks = retrieve_by_source("sales_forecast", top_n=20)
                    else:
                        source_filter = _detect_sources(msg)
                        chunks = retrieve(msg, top_k=3,
                                         source_filter=source_filter)
                        if not chunks and source_filter:
                            chunks = retrieve(msg, top_k=3,
                                             source_filter=None)

                    # Stream tokens
                    for token in call_ollama(msg, chunks, stream=True):
                        full_text += token
                        stream_placeholder.markdown(full_text + "▌")
                    stream_placeholder.markdown(full_text)

                    citations = "📊 *Sources: " + " · ".join(
                        dict.fromkeys(
                            c.get("source","").replace("_"," ").title()
                            for c in chunks
                        )
                    ) + "*"
                    st.markdown(citations)
                    st.caption("🤖 Llama · local inference · privacy-first")
                    mode_used = "rag"
                else:
                    raise ConnectionError("Ollama not available")

            except Exception as e:
                # Fallback to keyword matcher
                print(f"[chat] Streaming error: {e} — falling back")
                from modules.chatbot.response_logic import (
                    get_response as kw_response
                )
                payload_fb, _ = kw_response(
                    msg, st.session_state.data, st.session_state.ctx
                )
                full_text = payload_fb.get("text", "Sorry, I couldn't process that.")
                stream_placeholder.markdown(full_text)
                st.caption("🔤 Rule-based mode")
                mode_used = "keyword"
                chunks = []

        payload = {
            "text": full_text, "fig": None, "table": None,
            "image_path": None, "extra_images": None,
            "citations": "", "mode": mode_used, "chunks": [],
        }
        st.session_state.ctx = st.session_state.ctx
        st.session_state.messages.append({"role": "assistant", "payload": payload})
        st.rerun()

with tab_chat:
    # Empty state
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center;padding:40px 20px;color:#555;font-size:14px;">
                <div style="font-size:36px;margin-bottom:12px;">💬</div>
                <strong style="color:#888;">Start by asking a question</strong><br>
                Use the quick actions on the left, or type below.<br><br>
                <em>Examples: "what should I reorder?", "what are customers complaining about?",
                "which zone is busiest?"</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render message history
    for i, m in enumerate(st.session_state.messages):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar="🏪"):
                payload = m.get("payload", {})
                _render_payload(payload, msg_idx=i)
                citations = payload.get("citations", "")
                mode      = payload.get("mode", "")
                if citations:
                    st.markdown(citations)
                if mode == "rag":
                    st.caption("🤖 Llama 3.2 · local inference · privacy-first")
                elif mode == "keyword":
                    st.caption("🔤 Rule-based mode · start Ollama for AI responses")

    # Chat input — sets pending_message, triggers rerun above on next cycle
    user_text = st.chat_input("Ask anything about your retail data…")
    if user_text:
        st.session_state.pending_message = user_text
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ────────────────────────────────
# TAB 2 · Dashboard
# ────────────────────────────────
with tab_dashboard:
    import plotly.express as px
    import plotly.graph_objects as go
    data = st.session_state.data
    sales = data.get("sales")

    # ── Change 8: Product comparison chart ──
    st.subheader("Product sales comparison")
    if isinstance(sales, pd.DataFrame) and not sales.empty:
        all_pids = sorted(sales["product_id"].astype(str).unique().tolist())
        selected_pids = st.multiselect(
            "Select products to compare (actual vs forecast):",
            options=all_pids,
            default=all_pids[:2] if len(all_pids) >= 2 else all_pids,
            key="dashboard_product_select",
        )
        if selected_pids:
            df_sel = sales[sales["product_id"].astype(str).isin(selected_pids)].copy()
            df_sel["date"] = pd.to_datetime(df_sel["date"])
            df_sel = df_sel.sort_values("date")

            fig_cmp = go.Figure()
            colors  = px.colors.qualitative.Plotly
            for i, pid in enumerate(selected_pids):
                sub = df_sel[df_sel["product_id"].astype(str) == pid]
                c   = colors[i % len(colors)]
                # Confidence band
                if "predicted_low" in sub.columns and "predicted_high" in sub.columns:
                    fig_cmp.add_trace(go.Scatter(
                        x=pd.concat([sub["date"], sub["date"][::-1]]),
                        y=pd.concat([sub["predicted_high"], sub["predicted_low"][::-1]]),
                        fill="toself",
                        fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(c))+[0.12])}",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip",
                    ))
                # Actual
                if "actual_sales" in sub.columns:
                    fig_cmp.add_trace(go.Scatter(
                        x=sub["date"], y=sub["actual_sales"],
                        mode="lines", name=f"{pid} actual",
                        line=dict(color=c, width=2),
                    ))
                # Predicted
                pred_col = next(
                    (c2 for c2 in ["predicted_sales","yhat"] if c2 in sub.columns), None
                )
                if pred_col:
                    fig_cmp.add_trace(go.Scatter(
                        x=sub["date"], y=sub[pred_col],
                        mode="lines", name=f"{pid} forecast",
                        line=dict(color=c, width=1.5, dash="dash"),
                    ))

            fig_cmp.update_layout(
                title="Actual vs Forecast — Product Comparison",
                xaxis_title="Date", yaxis_title="Sales (£)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=20, l=0, r=0),
            )
            st.plotly_chart(fig_cmp, use_container_width=True, key="dashboard_comparison")
    else:
        st.info("Sales forecast not loaded. Run the Sales Forecasting module first.")

    col1, col2 = st.columns(2)

    # ── Inventory ──
    with col1:
        st.subheader("Inventory status")
        inv = data.get("inventory")
        if isinstance(inv, pd.DataFrame) and "decision" in inv.columns:
            counts = inv["decision"].value_counts().reset_index()
            counts.columns = ["decision", "count"]
            fig_inv = px.pie(
                counts, names="decision", values="count",
                color="decision",
                color_discrete_map={"REORDER": "#f59e0b", "OK": "#22c55e",
                                    "DISCOUNT": "#a78bfa"},
                title="Inventory decisions",
                template="plotly_dark", hole=0.45,
            )
            fig_inv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="DM Sans",
                margin=dict(t=40, b=10, l=0, r=0),
            )
            st.plotly_chart(fig_inv, use_container_width=True, key="dash_inv")
        else:
            st.info("Inventory data not loaded.")

    # ── Segments ──
    with col2:
        st.subheader("Customer segments")
        seg = data.get("segments")
        if isinstance(seg, pd.DataFrame) and "segment_name" in seg.columns:
            fig_seg = px.bar(
                seg, x="segment_name", y="customer_count",
                color="segment_name",
                title="Customers per segment",
                labels={"segment_name": "Segment", "customer_count": "Customers"},
                template="plotly_dark",
            )
            fig_seg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, font_family="DM Sans",
                margin=dict(t=40, b=20, l=0, r=0),
            )
            st.plotly_chart(fig_seg, use_container_width=True, key="dash_seg")
        else:
            st.info("Segment data not loaded.")

    # ── Foot traffic ──
    st.subheader("Foot traffic by zone")
    cv = data.get("cv")
    if isinstance(cv, pd.DataFrame) and "zone_id" in cv.columns:
        zone_avg = cv.groupby("zone_id")["people_count"].mean().reset_index()
        zone_avg.columns = ["zone_id", "avg_people"]
        fig_cv = px.bar(
            zone_avg, x="zone_id", y="avg_people",
            color="avg_people", color_continuous_scale="Reds",
            title="Average people count per zone",
            labels={"zone_id": "Zone", "avg_people": "Avg people"},
            template="plotly_dark",
        )
        fig_cv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font_family="DM Sans",
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_cv, use_container_width=True, key="dash_cv")
    else:
        st.info("Foot traffic data not loaded.")

    # ── Change 9: Review sentiment trend with date filter ──
    st.subheader("Review sentiment over time")
    ri = data.get("review_insights")
    if isinstance(ri, pd.DataFrame) and "date" in ri.columns:
        ri2 = ri.copy()
        ri2["date"] = pd.to_datetime(ri2["date"], errors="coerce")
        ri2 = ri2.dropna(subset=["date"]).sort_values("date")

        # Date range filter
        date_range = st.select_slider(
            "Date range",
            options=["All time", "Last 5 years", "Last 3 years",
                     "Last 2 years", "Last 1 year"],
            value="Last 2 years",
            key="dashboard_date_range",
        )
        cutoffs = {
            "Last 5 years": 5, "Last 3 years": 3,
            "Last 2 years": 2, "Last 1 year": 1,
        }
        if date_range in cutoffs:
            cutoff = ri2["date"].max() - pd.DateOffset(years=cutoffs[date_range])
            ri2 = ri2[ri2["date"] >= cutoff]

        rating_col = next(
            (c for c in ["avg_rating", "rating"] if c in ri2.columns), None
        )
        if rating_col and len(ri2) > 0:
            fig_ri = go.Figure()
            fig_ri.add_trace(go.Scatter(
                x=ri2["date"], y=ri2[rating_col],
                mode="lines+markers", name="Avg Rating",
                line=dict(color="#636EFA", width=2),
                yaxis="y1",
            ))
            if "positive_ratio" in ri2.columns:
                fig_ri.add_trace(go.Scatter(
                    x=ri2["date"], y=ri2["positive_ratio"] * 100,
                    mode="lines", name="Positive %",
                    line=dict(color="#00CC96", width=1.5, dash="dot"),
                    yaxis="y2",
                ))
            if "negative_ratio" in ri2.columns:
                fig_ri.add_trace(go.Scatter(
                    x=ri2["date"], y=ri2["negative_ratio"] * 100,
                    mode="lines", name="Negative %",
                    line=dict(color="#EF553B", width=1.5, dash="dot"),
                    yaxis="y2",
                ))
            fig_ri.update_layout(
                title=f"Sentiment trend — {date_range}",
                xaxis_title="Date",
                yaxis=dict(title="Avg Rating (1–5)", range=[0, 5]),
                yaxis2=dict(
                    title="Sentiment %", overlaying="y",
                    side="right", range=[0, 100],
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=20, l=0, r=0),
            )
            st.plotly_chart(fig_ri, use_container_width=True, key="dash_rating")
    else:
        st.info("Review data not loaded.")

# ────────────────────────────────
# TAB 3 · Setup
# ────────────────────────────────
with tab_setup:
    CONFIG_PATH = ROOT_DIR / "data" / "ui_paths.json"

    def _load_cfg() -> dict:
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cfg(cfg: dict):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _run_script(script_rel: str, args: list[str]) -> tuple[bool, str, str]:
        script_path = ROOT_DIR / script_rel
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR)
        p = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        return p.returncode == 0, p.stdout, p.stderr

    cfg = _load_cfg()
    defaults = {
        "cv_dir": cfg.get("cv_dir", ""),
        "sales_csv": cfg.get("online_retail_II.xlsx", "online_retail_II.xlsx"),
        "reviews_csv": cfg.get("reviews_csv", ""),
        "seg_csv": cfg.get("seg_csv", ""),
        "stock_csv": cfg.get("stock_csv", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Step 1: Paths
    st.markdown(
        '<div class="step-title"><span class="step-badge">1</span>Dataset paths</div>',
        unsafe_allow_html=True,
    )
    st.caption("Enter the absolute paths to your raw dataset files.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.text_input("CV zone folder (contains subfolders 1/, 2/, 41/…)", key="cv_dir")
        st.text_input("Sales CSV (POS transaction file)", key="sales_csv")
        st.text_input("Reviews CSV", key="reviews_csv")
    with col_b:
        st.text_input("Customer segmentation CSV", key="seg_csv")
        st.text_input("Stock CSV  (optional – auto-generates if blank)", key="stock_csv")
        st.write("")
        if st.button("💾  Save paths", use_container_width=True):
            _save_cfg(
                {k: st.session_state.get(k, "") for k in defaults}
            )
            st.success("Paths saved to data/ui_paths.json ✅")

    st.divider()

    # Step 2: Status
    st.markdown(
        '<div class="step-title"><span class="step-badge">2</span>Input / output status</div>',
        unsafe_allow_html=True,
    )
    inputs_exist = {
        "CV folder":        bool(st.session_state.get("cv_dir"))       and Path(st.session_state["cv_dir"]).exists(),
        "Sales CSV":        bool(st.session_state.get("sales_csv"))     and Path(st.session_state["sales_csv"]).exists(),
        "Reviews CSV":      bool(st.session_state.get("reviews_csv"))   and Path(st.session_state["reviews_csv"]).exists(),
        "Segmentation CSV": bool(st.session_state.get("seg_csv"))       and Path(st.session_state["seg_csv"]).exists(),
        "Stock CSV":        (not st.session_state.get("stock_csv"))     or Path(st.session_state["stock_csv"]).exists(),
    }

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Inputs**")
        for name, ok in inputs_exist.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} {name}")
    with col_s2:
        st.markdown("**Outputs (data/outputs/)**")
        for key, ok in outputs_status().items():
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} {key}")

    st.divider()

    # Step 3: Run modules
    st.markdown(
        '<div class="step-title"><span class="step-badge">3</span>Run AI modules</div>',
        unsafe_allow_html=True,
    )
    st.caption("Each button runs the corresponding module's output_generator.py using the paths above.")

    modules = [
        ("📈 Sales Forecast",       "modules/sales_forecasting/output_generator.py",    ["--input_excel", st.session_state.get("online_retail_II.xlsx","")]),
        ("📦 Inventory + Discount", "modules/inventory_management/output_generator.py", ["--sales_excel", st.session_state.get("online_retail_II.xlsx","")]),
        ("💬 Reviews Analysis",     "modules/reviews_analysis/output_generator.py",     ["--input_csv", st.session_state.get("Reviews.csv","")]),
        ("👥 Segmentation",         "modules/customer_segmentation/output_generator.py",["--input_excel", st.session_state.get("online_retail_II.xlsx","")]),
        ("🚶 CV Foot Traffic",      "modules/cv_foot_traffic/output_generator.py",      ["--input_dir", st.session_state.get("cv_dir","")]),
       
    ]

    run_cols = st.columns(3)
    for i, (label, script, args) in enumerate(modules):
        with run_cols[i % 3]:
            if st.button(label, use_container_width=True, key=f"run_{i}"):
                with st.spinner(f"Running {label}…"):
                    ok, out, err = _run_script(script, args)
                if ok:
                    st.success(f"{label} – Done ✅")
                else:
                    st.error(f"{label} – Failed ❌")
                if out:
                    with st.expander("stdout"):
                        st.code(out)
                if err:
                    with st.expander("stderr"):
                        st.code(err)

    st.write("")
    if st.button("↺  Refresh outputs after running", use_container_width=True):
        st.session_state.data = _refresh_data_cache()
        st.success("Outputs reloaded ✅")


# ────────────────────────────────
# TAB 4 · About
# ────────────────────────────────
with tab_about:
    st.markdown(
        """
        ## RetailMind – AI Decision Support System

        A fully offline, privacy-first AI system that translates raw retail data into 
        clear, actionable business decisions. No data ever leaves your machine.

        ---

        ### Architecture
        | Layer | Technology | Purpose |
        |---|---|---|
        | **Local LLM** | Llama 3.2 3B via Ollama | Natural language answers |
        | **RAG pipeline** | FAISS + MiniLM (all-MiniLM-L6-v2) | Semantic retrieval over retail data |
        | **Fallback** | Rule-based keyword matcher | Graceful degradation on limited hardware |
        | **UI** | Streamlit | Local web interface |

        ---

        ### AI modules
        | Module | Technique | Output |
        |---|---|---|
        | Computer Vision | YOLOv8n person detection | Zone foot-traffic counts + heatmaps |
        | Sales Forecasting | Facebook Prophet + confidence intervals | 30-day demand forecasts per product |
        | Customer Segmentation | K-Means on RFM features | Behavioural customer groups |
        | Reviews Analysis | VADER + TextBlob + LDA topic modeling | Complaint/praise themes (LDA) + sentiment trends |
        | Inventory Management | Dynamic safety stock + ABC classification | Reorder + discount recommendations |

        ---

        ### Datasets used
        | Module | Dataset | Size |
        |---|---|---|
        | Sales, Inventory & Segmentation | UCI Online Retail II (Kaggle) | 1M+ transactions |
        | Reviews & Sentiment | Amazon Fine Food Reviews (Kaggle) | 568K reviews |
        | CV Foot Traffic | Pretrained YOLOv8n (COCO person class) | — |

        ---

        ### Offline & privacy-first
        - **Local LLM**: Llama 3.2 3B runs entirely on your machine via Ollama
        - **Local embeddings**: MiniLM sentence transformer, no API calls
        - **Local vector store**: FAISS index stored on disk
        - **No cloud dependency**: works without internet after initial setup

        ---

        ### How to use
        1. **Setup tab** → configure dataset paths → run each AI module to generate outputs
        2. **Chat tab** → ask questions in plain English — AI mode when Ollama is running, rule-based fallback otherwise
        3. **Dashboard tab** → visual overview of all current outputs

        ---

        > Graduation project — Faculty of Computers and Artificial Intelligence, Benha University.
        > All business metrics derived from real public datasets. System runs 100% offline.
        """,
        unsafe_allow_html=True,
    )
