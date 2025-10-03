"""
High-level helper to interact with the compiled LangGraph agent.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .config import Settings, get_settings
from .graph import create_agent


class AgentPipeline:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.graph = create_agent(self.settings)

    def ask(self, question: str) -> Dict[str, Any]:
        """Invoke the graph with the provided question and return the resulting state."""
        initial_state: Dict[str, Any] = {"question": question}
        return self.graph.invoke(initial_state)

    def stream(self, question: str):
        """Yield intermediate states for streaming scenarios (used by Streamlit)."""
        initial_state: Dict[str, Any] = {"question": question}
        for event in self.graph.stream(initial_state):
            yield event


__all__ = ["AgentPipeline"]
