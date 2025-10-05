# LangGraph Weather + RAG Agent

This project implements an agentic AI pipeline that can either fetch live weather data from the OpenWeatherMap API or answer questions grounded in a PDF through Retrieval-Augmented Generation (RAG). The workflow is orchestrated with LangGraph, uses LangChain components for LLM and embeddings, stores vectors in Qdrant, and instruments LLM responses with LangSmith. A Streamlit chat UI is provided for an interactive demo.

## Features
- **LangGraph Router** automatically chooses between weather and RAG paths.
- **Weather integration** via the OpenWeatherMap REST API with structured summarisation.
- **PDF RAG pipeline** that loads, chunks, embeds, and stores documents in Qdrant.
- **LangChain LLMs/embeddings** (OpenAI by default) and LangSmith tracing hooks.
- **Streamlit chat UI** with debug insights into routing and retrieved context.
- **Pytest suite** that covers API handling, retrieval logic, and LLM response generation.

## Project Layout
```
+-- data/
�   +-- docs/            # Place your reference PDF here (see DOCUMENT_PATH)
+-- src/
�   +-- ai_pipeline/
�       +-- config.py    # Environment-driven settings
�       +-- graph.py     # LangGraph workflow and nodes
�       +-- llm.py       # LLM & embedding provider
�       +-- pipeline.py  # High-level helper for invoking the agent
�       +-- rag.py       # Qdrant-backed RAG services
�       +-- weather.py   # OpenWeatherMap client
+-- streamlit_app.py     # Streamlit interface
+-- tests/               # Pytest suite with unit tests
+-- requirements.txt     # Python dependencies
+-- .env.                # Template for required environment variables
```

## Prerequisites
- Python 3.10+
- OpenWeatherMap account and API key
- OpenAI API key (or update `llm.py` to use an alternative model provider)
- Qdrant instance (local Docker, managed, or `:memory:` for testing)
- LangSmith account for tracing (optional but recommended)

## Getting Started
1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   copy .env.example .env
   # edit .env in your editor and fill in the keys
   ```

   Required variables:
   - `OPENWEATHER_API_KEY`
   - `OPENAI_API_KEY`
   - `DOCUMENT_PATH` (path to the reference PDF for RAG)
   - `QDRANT_URL` / `QDRANT_API_KEY` (local URL or `:memory:` path for testing)
   - Optional LangSmith settings: `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`

3. **Prepare the document**
   Place the PDF you would like to query at the location defined by `DOCUMENT_PATH`. The default is `data/docs/reference.pdf`.

4. **Run the Streamlit demo**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the provided local URL, ask questions about the weather (e.g. "What is the weather in Paris today?") or about the PDF content.

## Programmatic Usage
You can invoke the agent directly from Python:
```python
from ai_pipeline.pipeline import AgentPipeline

pipeline = AgentPipeline()
result = pipeline.ask("Summarise the onboarding guide from the PDF")
print(result["answer"])
```

## Tests
The test suite validates the weather client, vector store retrieval, and LLM response logic.
```bash
pytest
```

> **Note:** Install the dependencies before running tests to avoid import errors. Tests use temporary Qdrant stores on disk and mocked/memory-backed services to remain fast and offline-friendly.

## LangSmith Tracing
Nodes in the LangGraph workflow are decorated with `@traceable` so that, when `LANGSMITH_API_KEY` is set, each run is automatically captured inside your LangSmith project. Use the LangSmith dashboard to review routing decisions, prompts, and model outputs. A Loom walkthrough demonstrating the traces should accompany this repository when submitting the assignment.

## Streamlit Demo Recording
Record a short Loom video showing:
- Launch of the Streamlit app
- Example weather question
- Example PDF-based answer
- LangSmith trace overview for one of the runs

## Troubleshooting
- **Missing API keys:** The Streamlit UI will stop and surface a helpful message if configuration is incomplete.
- **Qdrant connection issues:** Ensure the `QDRANT_URL` points to a reachable instance. For local testing, you can run `docker run -p 6333:6333 qdrant/qdrant` or set `QDRANT_URL=:memory:` to use a file-backed store.
- **PDF ingestion:** The first document query triggers ingestion if the collection is empty. For large PDFs, allow extra time for embedding generation.

Happy building!

### Test Reporting
Run all tests and emit an explainer report (including code references and skip notes) with:
```bash
python generate_test_report.py
```
You can pass additional pytest flags after the script name, for example:
```bash
python generate_test_report.py -- -k weather
```
