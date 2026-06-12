---
title: InsightEngine Multimodal RAG
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🔍 InsightEngine: Multimodal RAG & Unified Research Agent

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square)](https://github.com/chroma-core/chroma)
[![Groq](https://img.shields.io/badge/LLM-Groq-red?style=flat-square)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_3.5_Flash-blue?style=flat-square&logo=google)](https://aistudio.google.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io/)

---

🤗 **Live Demo:** [Hugging Face Space Demo](https://huggingface.co/spaces/Ankit1-7/multimodal-rag-research-assistant)  
👤 **Author:** [Ankit Yadav](https://github.com/Ankit1-7) · [LinkedIn Connection](https://www.linkedin.com/in/ankit-71-yadav/) · [Email](mailto:ankit712006@gmail.com)  

---

**InsightEngine** is a production-ready, dual-mode intelligence platform that combines **multimodal document reasoning (RAG)** with a **graph-orchestrated parallel search agent**. Designed to ingest complex academic/corporate papers and execute deep web exploration, the application integrates state-of-the-art LLMs, persistent multi-vector stores, and agentic workflows.

---

## 💡 Core Functionality

### 1. 📄 Multimodal Document Reasoning (RAG)
* **Ingestion**: Parses PDF documents to extract text, layout tables, and raw images.
* **Multi-Vector Indexing**: Employs SentenceTransformers for text chunks, utilizes Groq to generate table descriptions, and captions images/charts using Gemini Vision API before storing all vectors in **ChromaDB**.
* **Intent-Aware Routing**: Classifies user queries (Text vs. Tabular vs. Visual queries) and queries the target database modality.
* **Cited Synthesis**: Natively references context sources across text blocks, tables, and images.

### 2. 🔬 Graph-Orchestrated Research Agent
* **Refinement**: Uses Groq to extract search intents and expand topics.
* **Parallel Workflows**: Triggers 5 parallel scraping and api connectors:
  * **Academic**: ArXiv & Semantic Scholar APIs for recent papers.
  * **Literature**: Open Library & Google Books search.
  * **Code**: GitHub Search API for related repositories.
  * **Web**: Scrapes developer documentation and blog posts.
  * **Media**: YouTube Search API for tutorial and video coverage.
* **AI Executive Summary**: Gathers all results into a single state and synthesizes an executive summary via Gemini 3.5 Flash.

---

## 🏗️ Architecture & Workflows

### Multimodal RAG Pipeline
```
[PDF Upload] ──> pdfplumber & PyMuPDF (fitz)
                  ├── Text Blocks ──────> SentenceTransformer Embeddings ──> ChromaDB (text_chunks)
                  ├── Structural Tables ─> Groq Llama 3 Summary ───────────> ChromaDB (table_descriptions)
                  └── Images ───────────> Gemini 3.5 Vision Captions ─────> ChromaDB (image_captions)

[User Query] ──> Groq Query Router (Detects target modalities)
                  └── ChromaDB Search (Targeted collections)
                        └── Gemini 3.5 Flash (Cited Answer Synthesis)
```

### Research Agent LangGraph Workflow
```
[User Query] ──> Groq Topic Refiner
                   ├── Search Papers (ArXiv & Semantic Scholar APIs)
                   ├── Search Books (Open Library & Google Books)
                   ├── Search GitHub Repos (GitHub Search API)
                   ├── Search Web (Scraping Documentation & Medium)
                   └── Search Videos (YouTube API Scraper)
                        └── Gather & Rank ──> Gemini 3.5 Flash (Executive Synthesis)
```

---

## 🛠️ Key Technical Challenges Solved (Production Engineering)

To bring this project to a production-ready state, several integration hurdles were successfully debugged:

* **LangGraph State Merging (`InvalidUpdateError`)**: In parallel workflows (fan-out), returning full state dictionaries causes key collisions on merge (fan-in). We refactored parallel nodes in `src/rag_pipeline.py` to return only their specific output key modifications, allowing LangGraph to merge states safely.
* **API Version Lock & Deprecations**: 
  - Upgraded models from deprecated Gemini 1.5 to **Gemini 3.5 Flash** (supporting fast vision captioning and final answer generation).
  - Swapped decommissioned Groq Llama 3 models with **Llama 3.1 8B Instant** and **Llama 3.3 70B Versatile** to prevent API routing failures.
* **Client Constructor Conflicts**: Newer versions of `httpx` (0.28.0+) removed the `proxies` parameter in the client constructor, causing `ChatGroq` initializations to crash. Pinned `httpx==0.27.2` in `requirements.txt` to guarantee compatibility.
* **Platform Dependency Compilation**: Scientfic packages (like `numpy`, `chromadb`) fail compiler builds on Python 3.14. Resolved by locking local environments and Hugging Face builders to Python 3.11/3.12.
* **Telemetry Control**: Disabled anonymous telemetry logging in ChromaDB using environment flags to reduce terminal noise and speed up startup times.

---

## 📁 Repository Structure

```
InsightEngine/
├── app.py                     # Streamlit frontend application
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Files excluded from version control
├── .env.example               # Template environment variables
├── src/
│   ├── chroma_store.py        # ChromaDB setup and operations
│   ├── config.py              # System prompts and model names
│   ├── generator.py           # Gemini response synthesis
│   ├── indexer.py             # Ingestion and vector indexing
│   ├── llm_clients.py         # LLM client initializations
│   ├── multimodal_parser.py   # Text, table, and image extraction
│   ├── query_router.py        # Groq query router
│   ├── rag_pipeline.py        # LangGraph RAG pipeline orchestrator
│   ├── research_agent.py      # LangGraph Research Agent orchestrator
│   ├── retriever.py           # Vector database search retriever
│   ├── ui_components.py       # Render components for Streamlit
│   └── tools/                 # Connector scripts
│       ├── arxiv_tool.py      # ArXiv search connector
│       ├── book_tool.py       # Open Library book connector
│       ├── github_tool.py     # GitHub API repository connector
│       ├── website_tool.py    # Web documentation scraping connector
│       └── youtube_tool.py    # YouTube API search connector
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11 or 3.12
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ankit1-7/multimodal-rag-research-assistant.git
   cd multimodal-rag-research-assistant
   ```

2. Set up virtual environment (Python 3.11 or 3.12):
   ```bash
   py -3.11 -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment configuration:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY and GEMINI_API_KEY inside .env
   ```

5. Run the server:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Deployment Guide (Hugging Face Spaces)

This repository is pre-configured and optimized to run on **Hugging Face Spaces** using the Streamlit SDK.

1. Create a Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space SDK to **Streamlit** and choose **Python 3.11**.
3. Link your GitHub repository (`Ankit1-7/multimodal-rag-research-assistant`) to the Space in the settings page.
4. Under **Variables and secrets**, add the following environment variables:
   - `GROQ_API_KEY` (from console.groq.com)
   - `GEMINI_API_KEY` (from aistudio.google.com)
   - `GITHUB_TOKEN` *(optional)*
   - `YOUTUBE_API_KEY` *(optional)*
5. The build will launch automatically, and your app will be live!
