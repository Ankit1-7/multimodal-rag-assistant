"""
app.py — Multimodal RAG Research Assistant
==========================================
Streamlit application for HuggingFace Spaces deployment.

Two modes:
  1. 📄 RAG Chat — Upload PDF, ask questions, get multimodal answers
  2. 🔬 Research Explorer — Search papers, books, repos, videos on any ML/AI topic

Stack:
  • Groq (llama3-70b-8192) — Fast routing, table descriptions, query classification
  • Gemini (gemini-3.5-flash) — Final answer synthesis, image captioning
  • ChromaDB — Persistent vector store (text + image + table)
  • LangGraph — Orchestration for both RAG indexing and research pipelines
"""

import os
os.environ["ANON_TELEMETRY"] = "False"
import sys
import traceback
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure src is importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Create required directories ───────────────────────────────────────────────
for d in ["./data/chroma_db", "./data/uploads", "./data/extracted/images", "./data/extracted/tables"]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_keys() -> dict:
    """Load API keys from Streamlit secrets or session state inputs with safe env fallbacks."""
    # Try loading from environment variables first (as default values)
    env_groq = os.getenv("GROQ_API_KEY", "")
    env_gemini = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    env_github = os.getenv("GITHUB_TOKEN", "")
    env_youtube = os.getenv("YOUTUBE_API_KEY", "")

    keys = {
        "groq": st.session_state.get("groq_key_input", env_groq),
        "gemini": st.session_state.get("gemini_key_input", env_gemini),
        "github": st.session_state.get("github_key_input", env_github),
        "youtube": st.session_state.get("youtube_key_input", env_youtube),
    }
    # Override with Streamlit secrets if available
    try:
        if st.secrets.get("GROQ_API_KEY"):
            keys["groq"] = st.secrets["GROQ_API_KEY"]
        if st.secrets.get("GEMINI_API_KEY"):
            keys["gemini"] = st.secrets["GEMINI_API_KEY"]
        elif st.secrets.get("GOOGLE_API_KEY"):
            keys["gemini"] = st.secrets["GOOGLE_API_KEY"]
        if st.secrets.get("GITHUB_TOKEN"):
            keys["github"] = st.secrets["GITHUB_TOKEN"]
        if st.secrets.get("YOUTUBE_API_KEY"):
            keys["youtube"] = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return keys


def validate_keys(keys: dict) -> tuple[bool, str]:
    """Check that required API keys are present."""
    missing = []
    if not keys["groq"]:
        missing.append("Groq API Key")
    if not keys["gemini"]:
        missing.append("Gemini API Key")
    if missing:
        return False, f"Missing required keys: {', '.join(missing)}"
    return True, ""


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Apply modern typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.85) 50%, rgba(49, 16, 66, 0.95) 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        text-align: center;
        border: 1px solid rgba(129, 140, 248, 0.25);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .main-header h1 { 
        color: #f8fafc; 
        margin: 0; 
        font-size: 2.5rem; 
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p { color: #cbd5e1; margin: 0.6rem 0 0; font-size: 1.05rem; }
    
    .profile-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.4rem;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .profile-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.35);
    }
    .profile-card h3 { 
        margin: 0; 
        color: #818cf8; 
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .profile-card p {
        margin: 0.3rem 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .profile-badge {
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(129, 140, 248, 0.12);
        color: #c7d2fe;
        padding: 4px 9px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.3rem;
        border: 1px solid rgba(129, 140, 248, 0.25);
    }
    .profile-link {
        text-decoration: none;
        color: #94a3b8;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 6px;
        transition: all 0.2s ease;
    }
    .profile-link:hover {
        color: #a5b4fc;
        background: rgba(129, 140, 248, 0.08);
        border-color: rgba(129, 140, 248, 0.2);
        transform: translateX(2px);
    }
    .stat-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .answer-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #818cf8;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .section-header {
        border-bottom: 2px solid rgba(129, 140, 248, 0.3);
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.5rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        margin-bottom: 0.8rem !important;
        background: rgba(30, 41, 59, 0.2) !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
        background: linear-gradient(135deg, #5a52ef 0%, #8b4cf0 100%);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(keys: dict):
    with st.sidebar:
        # 1. Classy Profile Card (Personalized)
        st.markdown("## 👤 Developer Profile")
        st.markdown(
            f"""
            <div class="profile-card">
                <h3>Ankit Yadav</h3>
                <p><b> ECE Final Year student</b></p>
                <p style="font-size: 0.82rem; color: #94a3b8; display: flex; align-items: center; gap: 4px;">
                    📍 NSUT, Delhi
                </p>
                <p style="font-size: 0.82rem; line-height: 1.4; color: #cbd5e1; margin-top: 0.5rem; margin-bottom: 0.8rem;">
                    Building production-grade agentic AI systems, multimodal RAG search pipelines, and graph-orchestrated workflows.
                </p>
                <div style="margin-bottom: 0.8rem;">
                    <span class="profile-badge">Python</span>
                    <span class="profile-badge">LangGraph</span>
                    <span class="profile-badge">ChromaDB</span>
                    <span class="profile-badge">LLMs</span>
                </div>
                <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.6rem; display: flex; flex-direction: column; gap: 2px;">
                    <a href="https://github.com/Ankit1-7" target="_blank" class="profile-link">
                        💻 GitHub Profile
                    </a>
                    <a href="https://www.linkedin.com/in/ankit-71-yadav/" target="_blank" class="profile-link">
                        🔗 LinkedIn Connection
                    </a>
                    <a href="mailto:ankit712006@gmail.com" class="profile-link">
                        📧 Gmail
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. System Status (Minimal & Subtle)
        st.markdown("## ⚡ System Status")
        groq_ok = bool(keys.get("groq"))
        gemini_ok = bool(keys.get("gemini"))
        
        if groq_ok and gemini_ok:
            st.markdown(
                '<div style="display: flex; align-items: center; gap: 8px; color: #10b981; font-weight: 600; font-size: 0.9rem;">'
                '<span style="height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>'
                'System Secure & Active'
                '</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="display: flex; align-items: center; gap: 8px; color: #ef4444; font-weight: 600; font-size: 0.9rem;">'
                '<span style="height: 10px; width: 10px; background-color: #ef4444; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #ef4444;"></span>'
                'System Offline (Check Keys)'
                '</div>', 
                unsafe_allow_html=True
            )

        st.divider()
        
        # 3. About Project info
        st.markdown("## ⚙️ System Capabilities")
        st.markdown("""
**Document Intelligence (RAG)**
- Fully indexes PDFs across text, structural tables, and charts/images.
- Performs multi-vector search for context-aware querying.
- Natively generates cited references.

**Unified Research Agent**
- Conducts parallel real-time searches across papers, code repositories, books, and educational media.
- Provides deep, synthesized AI summaries.
        """)
        st.divider()
        st.caption("Built with LangGraph · ChromaDB · Streamlit")



# ── RAG Tab ───────────────────────────────────────────────────────────────────

def render_rag_tab(keys: dict):
    st.markdown('<h2 class="section-header">📄 Multimodal Document Q&A</h2>', unsafe_allow_html=True)
    st.markdown("Upload a PDF and ask questions. The system searches across **text, images, and tables**.")

    valid, err = validate_keys(keys)
    if not valid:
        st.warning(f"⚠️ {err}")

    uploaded_file = st.file_uploader(
        "📂 Upload PDF Document",
        type=["pdf"],
        help="Upload any PDF — reports, papers, books, manuals",
    )

    with st.expander("⚙️ Advanced Ingestion & Retrieval Settings"):
        col1, col2 = st.columns(2)
        with col1:
            skip_images = st.checkbox(
                "⏭️ Skip image captioning (Gemini Vision)",
                value=False,
                help="Speeds up indexing but skips diagrams and charts."
            )
            skip_tables = st.checkbox(
                "⏭️ Skip table extraction",
                value=False,
                help="Speeds up indexing but skips tabular content."
            )
        with col2:
            retrieval_k = st.slider(
                "Search context size (k)",
                min_value=1,
                max_value=6,
                value=3,
                help="Number of relevant chunks to retrieve per modality."
            )

    if uploaded_file is not None:
        doc_name = Path(uploaded_file.name).stem

        # Save uploaded file
        upload_path = f"./data/uploads/{uploaded_file.name}"
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Index button
        already_indexed = st.session_state.get(f"indexed_{doc_name}", False)
        index_label = "✅ Re-index Document" if already_indexed else "🔄 Index Document"

        col_btn1, _ = st.columns([2, 3])
        with col_btn1:
            do_index = st.button(index_label, type="primary", use_container_width=True)

        if do_index:
            if not valid:
                st.error(f"Cannot index: {err}")
            else:
                with st.spinner("🔄 Parsing and indexing document... (may take a minute)"):
                    try:
                        from src.rag_pipeline import index_document
                        stats = index_document(
                            file_path=upload_path,
                            doc_name=doc_name,
                            groq_api_key=keys["groq"],
                            gemini_api_key=keys["gemini"],
                            skip_images=skip_images,
                            skip_tables=skip_tables,
                        )
                        st.session_state[f"indexed_{doc_name}"] = True
                        st.session_state["current_doc"] = doc_name

                        if stats.get("error"):
                            st.warning(f"⚠️ Partial index: {stats['error']}")

                        st.success(f"✅ Document indexed successfully!")
                        from src.ui_components import render_index_stats
                        render_index_stats(
                            stats.get("text_count", 0),
                            stats.get("image_count", 0),
                            stats.get("table_count", 0),
                        )
                    except Exception as e:
                        st.error(f"❌ Indexing failed: {e}")
                        st.code(traceback.format_exc())

        # Q&A section
        st.divider()
        if already_indexed or st.session_state.get(f"indexed_{doc_name}"):
            st.markdown("### 💬 Ask Questions")
            st.caption(f"Querying: **{doc_name}**")

            # Chat history
            if "rag_messages" not in st.session_state:
                st.session_state["rag_messages"] = []

            # Display history
            for msg in st.session_state["rag_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input
            query = st.chat_input("Ask anything about the document...")
            if query:
                if not valid:
                    st.error(f"Cannot answer: {err}")
                else:
                    st.session_state["rag_messages"].append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Searching across text, images, and tables..."):
                            try:
                                from src.rag_pipeline import answer_query
                                result = answer_query(
                                    query=query,
                                    groq_api_key=keys["groq"],
                                    gemini_api_key=keys["gemini"],
                                )
                                answer = result["answer"]
                                qt = result.get("query_types", [])
                                retrieved = result.get("retrieved_results", [])

                                # Display modality badges
                                if qt:
                                    badges = " ".join(f"`{t}`" for t in qt)
                                    st.caption(f"🎯 Searched modalities: {badges}")

                                st.markdown(answer)

                                # Show retrieved context
                                if retrieved:
                                    with st.expander("🔎 View Reference Sources & Citations"):
                                        from src.ui_components import render_retrieved_context
                                        render_retrieved_context(retrieved)

                                st.session_state["rag_messages"].append({
                                    "role": "assistant",
                                    "content": answer
                                })

                            except Exception as e:
                                err_msg = f"❌ Error: {e}"
                                st.error(err_msg)
                                st.code(traceback.format_exc())
                                st.session_state["rag_messages"].append({
                                    "role": "assistant",
                                    "content": err_msg
                                })

            # Clear chat button
            if st.session_state.get("rag_messages"):
                if st.button("🗑️ Clear Chat History"):
                    st.session_state["rag_messages"] = []
                    st.rerun()
        else:
            st.info("👆 Upload a PDF and click **Index Document** to begin.")


# ── Research Tab ──────────────────────────────────────────────────────────────

def render_research_tab(keys: dict):
    st.markdown('<h2 class="section-header">🔬 ML/AI Research Explorer</h2>', unsafe_allow_html=True)
    st.markdown("Search for the best papers, books, repos, and videos on any ML/AI/Data Science topic.")

    # ── Search form ───────────────────────────────────────────────────────────
    with st.form("research_form"):
        query = st.text_input(
            "🔍 Enter Research Topic or Question",
            placeholder="e.g. 'Transformer attention mechanisms', 'Graph Neural Networks for drug discovery'",
            help="Be specific for better results. ML/AI/DS topics work best.",
        )

        with st.expander("⚙️ Customize Search Focus & Limits"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Priority References** *(Optional)*")
                priority_channel = st.text_input(
                    "📺 Preferred YouTube Channel",
                    placeholder="e.g. 3Blue1Brown, Yannic Kilcher",
                    help="Prioritise videos from this channel",
                )
                priority_repo = st.text_input(
                    "💻 Preferred GitHub Repository",
                    placeholder="e.g. https://github.com/huggingface/transformers",
                    help="Prioritise this repository in results",
                )
                priority_paper = st.text_input(
                    "📄 Preferred Paper/Author Key",
                    placeholder="e.g. 'Attention is All You Need' or arxiv link",
                    help="Prioritise this paper or author",
                )

            with col2:
                st.markdown("**Search Limits**")
                max_papers = st.slider("Research papers to parse", 5, 20, 10)
                max_repos = st.slider("Repositories to analyze", 5, 20, 10)
                max_books = st.slider("Reference books to search", 3, 10, 5)
                top_display = st.slider("Initial results display count", 3, 10, 5)

        submitted = st.form_submit_button("🚀 Execute Research Agent", type="primary", use_container_width=True)

    # ── Run research ──────────────────────────────────────────────────────────
    if submitted and query.strip():
        valid, err = validate_keys(keys)
        if not valid:
            st.warning(f"⚠️ {err} — some features (AI summary) will be limited.")

        progress_bar = st.progress(0, text="Starting research pipeline...")
        status_text = st.empty()

        try:
            from src.research_agent import run_research

            status_text.text("🔍 Refining query with Groq...")
            progress_bar.progress(10, text="Refining query...")

            result = None
            with st.spinner("⚙️ Running full research pipeline (LangGraph)..."):
                result = run_research(
                    query=query.strip(),
                    groq_api_key=keys["groq"],
                    gemini_api_key=keys["gemini"],
                    github_token=keys["github"],
                    youtube_api_key=keys["youtube"],
                    priority_channel=priority_channel,
                    priority_repo_url=priority_repo,
                    priority_paper_url=priority_paper,
                    max_papers=max_papers,
                    max_books=max_books,
                    max_repos=max_repos,
                )

            progress_bar.progress(100, text="✅ Research complete!")
            status_text.empty()

            if result is None:
                st.error("Research pipeline returned no results.")
                return

            # ── Overview ──────────────────────────────────────────────────────
            refined_topic = result.get("topic", query)
            st.markdown(f"## 📊 Research Results: *{refined_topic}*")

            # Stats row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📄 Papers", len(result.get("papers", [])))
            c2.metric("📚 Books", len(result.get("books", [])))
            c3.metric("💻 Repos", len(result.get("repos", [])))
            c4.metric("🌐 Websites", len(result.get("websites", [])))
            c5.metric("▶️ Videos", len(result.get("videos", [])))

            # ── AI Summary ────────────────────────────────────────────────────
            if result.get("summary"):
                st.markdown("### 🤖 AI Research Summary")
                st.info(result["summary"])

            st.divider()

            # ── Tabbed results ────────────────────────────────────────────────
            from src.ui_components import (
                render_papers_section, render_books_section,
                render_repos_section, render_websites_section,
                render_videos_section,
            )

            tab_papers, tab_books, tab_repos, tab_websites, tab_videos = st.tabs([
                f"📄 Papers ({len(result.get('papers', []))})",
                f"📚 Books ({len(result.get('books', []))})",
                f"💻 Repos ({len(result.get('repos', []))})",
                f"🌐 Websites ({len(result.get('websites', []))})",
                f"▶️ Videos ({len(result.get('videos', []))})",
            ])

            with tab_papers:
                st.markdown("#### Research Papers")
                st.caption("Sources: ArXiv + Semantic Scholar · Ranked by citations + relevance")
                render_papers_section(result.get("papers", []), top_n=top_display)

            with tab_books:
                st.markdown("#### Books & Learning Materials")
                st.caption("Curated free books prioritised · Also includes Open Library + Google Books")
                render_books_section(result.get("books", []), top_n=top_display)

            with tab_repos:
                st.markdown("#### GitHub Repositories")
                st.caption("Official repos prioritised · Ranked by stars, forks, relevance")
                render_repos_section(result.get("repos", []), top_n=top_display)

            with tab_websites:
                st.markdown("#### Websites & Online Resources")
                st.caption("Documentation, courses, blogs, and tools")
                render_websites_section(result.get("websites", []), top_n=8)

            with tab_videos:
                st.markdown("#### YouTube Educational Videos")
                st.caption("Educational content · Priority channel shown first if specified")
                render_videos_section(result.get("videos", []), top_n=top_display)

            # ── Save to session ───────────────────────────────────────────────
            st.session_state["last_research"] = result
            st.session_state["last_query"] = query

        except Exception as e:
            progress_bar.progress(0)
            status_text.empty()
            st.error(f"❌ Research pipeline error: {e}")
            with st.expander("🔍 Debug Info"):
                st.code(traceback.format_exc())

    elif submitted and not query.strip():
        st.warning("Please enter a search query.")

    # ── Show last results if available ────────────────────────────────────────
    elif not submitted and st.session_state.get("last_research"):
        st.info(f"💾 Showing previous results for: **{st.session_state.get('last_query', 'Unknown')}**")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Multi-Modal Research Intelligence</h1>
        <p>Advanced Knowledge Discovery & Document Reasoning System</p>
        <p style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0.4rem;">
            Query complex documents across text, tables, and images, or explore unified academic & code research.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load keys
    keys = get_api_keys()

    # Sidebar
    render_sidebar(keys)

    # Main tabs
    tab_rag, tab_research = st.tabs([
        "📄 Document Q&A (RAG)",
        "🔬 Research Explorer",
    ])

    with tab_rag:
        render_rag_tab(keys)

    with tab_research:
        render_research_tab(keys)


if __name__ == "__main__":
    main()
