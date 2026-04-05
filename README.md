# Document Parser using RAG

This repository contains an AI agent application in the `Document_Parser` directory.

## Repository Layout

- `Document_Parser/`: Main application code, dependencies, and app-level documentation
- `README.md`: Repository-level overview and quick-start entry point

## Why the App Is in `Document_Parser`

The app is kept in a dedicated folder so project files (code, lockfile, environment config, and local artifacts) stay grouped together. This makes the repository root cleaner and keeps setup/running commands scoped to one place.

## Main Project

- App code and detailed documentation: `Document_Parser/`
- Project README: `Document_Parser/README.md`

## Quick Start

```bash
cd Document_Parser
uv sync
streamlit run rag_frontend_st.py
```

The app is a tool-calling agent that can use:

- PDF RAG retrieval (`rag_tool`)
- Internet search (DuckDuckGo)
