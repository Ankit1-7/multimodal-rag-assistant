"""
config.py — Central configuration for the Multimodal RAG Research Assistant.
Reads API keys from environment / Streamlit secrets with safe fallbacks.
"""

import os
import streamlit as st


def get_config() -> dict:
    """Load all configuration from env vars or st.secrets."""
    config = {}

    def load_key(name: str, fallback_env: str = "") -> str:
        val = ""
        try:
            val = st.secrets.get(name, "")
        except Exception:
            pass
        if not val:
            val = os.getenv(name, "")
        if not val and fallback_env:
            try:
                val = st.secrets.get(fallback_env, "")
            except Exception:
                pass
            if not val:
                val = os.getenv(fallback_env, "")
        return val

    config["groq_api_key"] = load_key("GROQ_API_KEY")
    config["gemini_api_key"] = load_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    config["github_token"] = load_key("GITHUB_TOKEN")
    config["youtube_api_key"] = load_key("YOUTUBE_API_KEY")

    return config


# Model settings
GROQ_MODEL = "llama-3.3-70b-versatile"   # Powerful, 14k req/day free
GROQ_FAST_MODEL = "gpt-oss-20b" # Fast model for routing/classification
GEMINI_MODEL = "gemini-3.5-flash"       # For final answer generation

# ChromaDB settings
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_TEXT = "text_chunks"
COLLECTION_IMAGES = "image_captions"
COLLECTION_TABLES = "table_descriptions"

# Extraction dirs
IMAGES_DIR = "./data/extracted/images"
TABLES_DIR = "./data/extracted/tables"
UPLOADS_DIR = "./data/uploads"

# Search result limits
MAX_PAPERS = 20
MAX_BOOKS = 10
MAX_REPOS = 20
TOP_DISPLAY = 5

EMBED_MODEL = "all-MiniLM-L6-v2"
