import os
import streamlit as st
from rag_engine import GrowwRAGEngine, GROWW_EDU_URL

# --- Page Configuration ---
st.set_page_config(
    page_title="Groww Mutual Fund FAQ Bot",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# Paste right here
st.markdown("""
<style>
    .stChatInput input {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Minimalist Dark Theme Styling (Teal/Green Accent #00D09C) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Foundation */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #0D1117 !important;
        color: #F0F6FC !important;
    }

    .stApp {
        background-color: #0D1117 !important;
    }

    /* Single-Column Spacing */
    .block-container {
        max-width: 740px !important;
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Hide Sidebar & Streamlit Header */
    section[data-testid="stSidebar"], #MainMenu, footer, header {
        display: none !important;
    }

    /* Header & Branding */
    .header-container {
        text-align: left;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 4px 0;
    }
    .brand-accent {
        color: #00D09C;
    }
    .brand-white {
        color: #F0F6FC;
    }
    .brand-subtitle {
        color: #8B949E;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }
    .brand-description {
        color: #C9D1D9;
        font-size: 0.94rem;
        line-height: 1.55;
        margin-bottom: 16px;
    }

    /* Trust Checkmarks & Disclaimer Badge */
    .trust-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 18px;
    }
    .trust-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.84rem;
        font-weight: 500;
        color: #E6EDF3;
    }
    .check-icon {
        color: #00D09C;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-disclaimer {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #FBBF24;
        background: rgba(245, 158, 11, 0.08);
        border: 1px dashed rgba(245, 158, 11, 0.55);
        padding: 3px 10px;
        border-radius: 9999px;
    }

    /* Covering Schemes Section */
    .covering-container {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 4px;
    }
    .covering-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8B949E;
        margin-right: 4px;
    }
    .scheme-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #C9D1D9;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 9999px;
        transition: border-color 0.2s ease;
    }
    .scheme-pill:hover {
        border-color: #00D09C;
        color: #00D09C;
    }

    /* Suggested Question Chips (3 max) */
    .suggestion-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B949E;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #E6EDF3 !important;
        border-radius: 9999px !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        text-align: left !important;
        width: 100% !important;
        min-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    div.stButton > button:hover {
        background: rgba(0, 208, 156, 0.08) !important;
        border-color: #00D09C !important;
        color: #00D09C !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 208, 156, 0.16) !important;
    }

    /* Chat Messages Layout */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 1.1rem 0.2rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    div[data-testid="stChatMessageContent"] {
        color: #F0F6FC !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }
    div[data-testid="stChatMessageContent"] a {
        color: #00D09C !important;
        font-weight: 600;
        text-decoration: none;
    }
    div[data-testid="stChatMessageContent"] a:hover {
        text-decoration: underline;
    }

    /* Frosted Bottom Input Bar */
    div[data-testid="stChatInput"] {
        padding-bottom: 1.5rem !important;
    }
    .stChatInputContainer {
        background: rgba(22, 27, 34, 0.88) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #00D09C !important;
        box-shadow: 0 0 0 1px #00D09C, 0 8px 24px rgba(0, 208, 156, 0.2) !important;
    }
    .stChatInputContainer textarea {
        color: #F0F6FC !important;
        font-size: 0.94rem !important;
    }

    /* Collapsible About Expander */
    div[data-testid="stExpander"] {
        background: rgba(22, 27, 34, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 12px !important;
        margin-top: 1.5rem !important;
    }
    div[data-testid="stExpander"] summary {
        color: #8B949E !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize RAG Engine ---
@st.cache_resource
def get_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sources_path = os.path.join(base_dir, "sources.csv")
    corpus_path = os.path.join(base_dir, "data", "corpus.json")
    return GrowwRAGEngine(sources_path, corpus_path)

engine = get_engine()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# --- Clean Header & Branding ---
st.markdown(
    """
    <div class="header-container">
        <h1 class="brand-title">
            <span class="brand-accent">Groww</span> <span class="brand-white">Mutual Fund FAQ Bot</span>
        </h1>
        <div class="brand-subtitle">AI-powered · Source-grounded answers.</div>
        <p class="brand-description">
            Ask factual questions about selected Groww mutual fund schemes. Every answer is grounded in official sources and includes a citation link.
        </p>
        <div class="trust-row">
            <span class="trust-item"><span class="check-icon">✓</span> Facts only</span>
            <span class="trust-item"><span class="check-icon">✓</span> Official sources</span>
            <span class="trust-item"><span class="check-icon">✓</span> No investment advice</span>
            <span class="badge-disclaimer">⚠ Not investment advice</span>
        </div>
        <div class="covering-container">
            <span class="covering-label">COVERING</span>
            <span class="scheme-pill">Groww Large Cap Fund</span>
            <span class="scheme-pill">Groww ELSS Tax Saver</span>
            <span class="scheme-pill">Groww Flexi Cap Fund</span>
            <span class="scheme-pill">SBI Bluechip Fund</span>
            <span class="scheme-pill">SBI Long Term Equity</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

def render_assistant_response(response: dict, msg_idx: int = 0):
    """Renders the exhaustive research output with Detailed Overview, Exhaustive Sources, and interactive Suggested Questions."""
    # 1. Detailed Overview
    overview_text = response.get("overview")
    if overview_text and response.get("status") == "ok":
        st.markdown("### Detailed Overview")
        st.markdown(overview_text)
    else:
        answer_text = response.get("plain_answer") or response.get("answer", "No answer found.")
        st.markdown(answer_text)

    # 2. Exhaustive Sources Section
    sources = response.get("sources", [])
    if sources and response.get("status") == "ok":
        st.markdown(f"**↗ EXHAUSTIVE SOURCES ({len(sources)} References)**")
        for source in sources:
            if isinstance(source, dict):
                title = source.get("title", "Official Source")
                url = source.get("url", GROWW_EDU_URL)
                org = source.get("organization", "Groww")
                st.link_button(f"↗ {title} · {org}", url)
            else:
                st.markdown(f"- ↗ **{source}**")
    elif response.get("url") and response.get("status") == "ok":
        st.markdown("**↗ EXHAUSTIVE SOURCES**")
        st.link_button(f"↗ {response.get('source', 'Official Source')}", response.get("url", GROWW_EDU_URL))

    # 3. Metadata (Last updated)
    last_updated = response.get("last_updated")
    if last_updated:
        st.caption(f"Last updated from sources · {last_updated}")

    # 4. Multi-Chunk Debug Expander
    with st.expander("Retrieved chunks & Data Points (debug)"):
        raw_chunks = response.get("raw_chunks", [])
        if raw_chunks:
            for idx, chunk in enumerate(raw_chunks, 1):
                st.markdown(f"**Chunk #{idx}: [{chunk.get('title')}]({chunk.get('url')})** (Relevance Score: `{chunk.get('score')}`)")
                st.caption(f"> {chunk.get('content')}")
        else:
            raw_debug_text = response.get("raw_text") or response.get("text") or "No raw text available."
            st.caption(f"**Retrieved Corpus Context:**\n{raw_debug_text}")
            if response.get("score") is not None:
                st.caption(f"**Relevance Similarity Score:** `{response.get('score')}`")

    # 5. Dynamic Suggested Questions (Interactive suggestion chips right after each answer)
    suggestions = response.get("suggestions", [])
    if not suggestions:
        answer = response.get("answer", "")
        if "### Suggested Questions" in answer:
            parts = answer.split("### Suggested Questions")[-1]
            suggestions = [line.strip("- ").strip() for line in parts.splitlines() if line.strip().startswith("-")]

    if suggestions and response.get("status") == "ok":
        st.markdown("<div style='margin-top: 14px; margin-bottom: 8px; font-weight: 600; font-size: 0.85rem; color: #8B949E;'>💡 Suggested Questions</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for s_idx, sug in enumerate(suggestions[:4]):
            col = cols[s_idx % 2]
            with col:
                if st.button(f"👉 {sug}", key=f"sug_btn_{msg_idx}_{s_idx}", use_container_width=True):
                    st.session_state.pending_query = sug
                    st.rerun()


# --- Render Chat Stream ---
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.get("content", ""))
    else:
        with st.chat_message("assistant", avatar="📈"):
            resp = msg.get("response")
            if resp and isinstance(resp, dict):
                render_assistant_response(resp, msg_idx=idx)
            else:
                st.markdown(msg.get("content", ""))

# --- Suggested Question Chips (Default above the input) ---
st.markdown("<div class='suggestion-label'>Suggested Questions</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏷️ ELSS 3-Yr Lock-in & 80C Tax Benefit", use_container_width=True):
        st.session_state.pending_query = "What is the lock-in period and tax benefit of an ELSS fund?"
        st.rerun()

with col2:
    if st.button("⏰ Cut-off Timings for Same-Day NAV", use_container_width=True):
        st.session_state.pending_query = "What are the cut-off timings for mutual fund purchases to get same-day NAV?"
        st.rerun()

with col3:
    if st.button("📊 Large-Cap vs Flexi-Cap SEBI Rules", use_container_width=True):
        st.session_state.pending_query = "What is the difference between Large-Cap and Flexi-Cap mutual funds under SEBI rules?"
        st.rerun()

# --- Single Centered Chat Input ---
user_input = st.chat_input("Ask about these mutual funds...")

active_query = None
if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif user_input:
    active_query = user_input

if active_query:
    # 1. Append User Input
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_query)

    # 2. Execute RAG Retrieval safely with try-except
    with st.chat_message("assistant", avatar="📈"):
        try:
            with st.spinner("Retrieving official documentation..."):
                response = engine.answer_query(active_query)
            
            # Display safely using helper
            render_assistant_response(response, msg_idx=len(st.session_state.messages))

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("answer", ""),
                "response": response
            })
        except Exception as e:
            st.error(f"An error occurred: {e}")
    st.rerun()

# --- Collapsible About Expander (Below input, closed by default) ---
with st.expander("ℹ️ About this bot", expanded=False):
    st.markdown(
        """
        **Official Regulatory & Scheme Grounding**
        - **Source Transparency**: Every response is extracted exclusively from indexed official public pages of Groww, SEBI, AMFI, and SBI Mutual Fund.
        - **Zero Hallucination Guardrail**: If relevant facts are not found in the official corpus above our confidence threshold, the bot strictly replies: *"I don't have this information in my official sources."*
        - **Strict Conciseness**: Responses are capped at a maximum of 3 sentences.
        - **Privacy & Safety Shield**: Automatically rejects requests containing personal identifying information (PAN, Aadhaar, phone numbers, or email) and politely declines investment advice or personal portfolio recommendations.
        """
    )

    st.markdown("---")
    st.markdown(f"**Indexed Official Filings:** {len(engine.sources)} URLs")
    for s in engine.sources.values():
        st.markdown(f"- [{s['organization']}: {s['title']}]({s['url']}) · *{s['category']}*")

    st.markdown("---")
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()
