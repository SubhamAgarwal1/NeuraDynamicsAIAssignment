from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from ai_pipeline.config import get_settings
from ai_pipeline.pipeline import AgentPipeline

st.set_page_config(page_title="LangGraph RAG + Weather Agent", page_icon="AI")

st.title("LangGraph Agent Demo")
st.caption("Ask about the weather or anything from your reference PDF")

settings = get_settings()

with st.sidebar:
    st.subheader("Configuration")
    st.markdown(
        """
        - Ensure `.env` contains your API keys.
        - Upload a PDF to `data/docs` or update `DOCUMENT_PATH`.
        - Weather data comes from OpenWeatherMap.
        """
    )
    st.code(
        f"Document path: {settings.document_path}\nCollection: {settings.qdrant_collection}",
        language="text",
    )

try:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = AgentPipeline(settings)
except ValueError as err:
    st.error(f"Configuration error: {err}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me about the weather or what's in the PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = st.session_state.pipeline.ask(prompt)
            answer = result.get("answer", "I could not produce an answer.")
            placeholder.markdown(answer)
            route = result.get("route", "unknown")
            with st.expander("Debug info"):
                st.json({
                    "route": route,
                    "weather": result.get("weather_payload"),
                    "context_documents": result.get("context_documents"),
                })
        except Exception as exc:
            answer = f"?? An error occurred: {exc}"
            placeholder.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
