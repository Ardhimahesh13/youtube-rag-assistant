# 🎬 YouTube RAG Assistant

### 🚀 Live Demo

**[👉 Open YouTube RAG Assistant](https://youtube-rag-assistant-etrzh3xhmmpmbu7phnrrg3.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://youtube-rag-assistant-etrzh3xhmmpmbu7phnrrg3.streamlit.app/)

An interactive Retrieval-Augmented Generation (RAG) application that lets users ask questions about YouTube videos and receive answers grounded in their transcripts.

### 🛠️ Tech Stack

- **Python** — Application logic
- **Streamlit** — Web interface and deployment
- **LangChain** — RAG pipeline orchestration
- **Hugging Face** — Text embeddings (`all-MiniLM-L6-v2`)
- **FAISS** — Vector storage and similarity search
- **Groq** — Hosted large language model (`openai/gpt-oss-20b`)
- **YouTube Transcript API** — Transcript extraction

### ✨ Features

- Extracts English captions from YouTube videos.
- Splits transcripts into manageable text chunks.
- Converts text into vector embeddings.
- Retrieves relevant transcript sections using FAISS.
- Generates answers using a Groq-hosted LLM.
- Displays retrieved transcript excerpts as sources.

### ▶️ How to Use

1. Open the [Live Demo](https://youtube-rag-assistant-etrzh3xhmmpmbu7phnrrg3.streamlit.app/).
2. Paste a YouTube video URL.
3. Click **Process video**.
4. Enter a question about the video.
5. Review the generated answer and retrieved transcript sources.

### 🏗️ RAG Architecture

YouTube URL → Transcript Extraction → Text Chunking → Hugging Face Embeddings → FAISS Retrieval → Groq LLM → Answer with Sources

### 🔐 Security

The Groq API key is configured using Streamlit Secrets and should never be committed to the public repository.

### 👨‍💻 Author

**Mahesh**

