# Document Parser using RAG and Web Search

This project is an AI agent built with LangGraph. It can decide between tools during a chat:

- RAG tool for answering from an uploaded PDF
- Internet search tool (DuckDuckGo) for web-based questions

The app includes a Streamlit UI with multi-thread chat sessions and per-thread document indexing.

## How the Agent Works

1. User sends a message in the Streamlit chat.
2. The agent reasons and decides whether to:
	- call `rag_tool` for uploaded-document context
	- call DuckDuckGo search for internet context
	- answer directly without tools
3. Tool outputs are fed back to the agent.
4. The agent returns a final response.

Chat state is checkpointed in a local SQLite database (`chatbot.db`) so thread conversations can be restored.

## Features

- PDF ingestion and chunking
- FAISS vector retrieval per chat thread
- Tool-calling AI agent (RAG + web search)
- Streamlit chat interface with thread history
- Local checkpoint persistence via SQLite

## Project Structure

- `rag_backend.py`: Agent graph, tools, PDF ingestion, retriever state, checkpointer
- `rag_frontend_st.py`: Streamlit UI, thread selection, file upload, chat streaming
- `pyproject.toml`: Project metadata and dependencies
- `.gitignore`: Ignores local environment/runtime files

## Requirements

- Python 3.12+
- OpenAI API key

## Setup

### Option 1: uv (recommended)

```bash
uv sync
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Variables

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

From this directory, run:

```bash
streamlit run rag_frontend_st.py
```

Open the local Streamlit URL shown in terminal output.

## Usage

1. Start a chat.
2. Upload a PDF from the sidebar to enable document-grounded answers in that thread.
3. Ask questions.
4. The agent will use RAG for PDF questions and internet search when web context is useful.

## Notes

- Keep secrets in `.env`; do not commit credentials.
- `.gitignore` already excludes `.env`, virtualenvs, caches, and `chatbot.db`.
