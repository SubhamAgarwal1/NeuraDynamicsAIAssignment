"""
LangGraph workflow that routes between weather lookup and RAG answers.
"""
from __future__ import annotations

import re
import requests
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .config import Settings, get_settings
from .llm import LLMProvider
from .logger import get_logger
from .rag import RAGService, VectorStoreManager
from .weather import WeatherService

try:  # LangSmith instrumentation is optional
    from langsmith import traceable
except ImportError:  # pragma: no cover - safety fallback
    def traceable(*_args, **_kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

LOGGER = get_logger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["weather", "rag"]
    weather_summary: str
    weather_payload: Dict[str, object]
    context_documents: List[str]
    answer: str


class RouteDecider:
    """Simple keyword based router to choose between weather or document search."""

    WEATHER_KEYWORDS = {"weather", "temperature", "forecast", "rain", "sun", "wind"}

    def __call__(self, state: AgentState) -> AgentState:
        question = state.get("question", "").lower()
        if any(word in question for word in self.WEATHER_KEYWORDS):
            route: Literal["weather", "rag"] = "weather"
        else:
            route = "rag"
        LOGGER.info("Routing question '%s' to %s", question, route)
        return {"route": route}






class WeatherNode:
    MAX_CANDIDATES = 6
    MAX_DOC_CANDIDATES = 4
    NEGATION_TOKENS = {"not", "isnt", "isn't", "no", "never"}
    LOCATION_KEYWORDS = {
        "india",
        "city",
        "state",
        "district",
        "kolkata",
        "howrah",
        "uttar",
        "pradesh",
        "noida",
        "greater",
        "bangalore",
        "karnataka",
        "chennai",
        "delhi",
        "mumbai",
        "pune",
        "hyderabad",
        "goa",
        "bengal",
        "maharashtra",
        "gujarat",
        "tamil",
        "nadu",
        "kerala",
        "andhra",
        "telangana",
        "lucknow",
        "agra",
        "jaipur",
        "gurgaon",
        "gurugram",
        "ahmedabad",
        "surat",
        "indore",
        "nagpur",
        "bhopal",
        "visakhapatnam",
        "varanasi",
        "patna",
        "bhubaneswar",
    }
    KNOWN_LOCATIONS = {
        "kolkata",
        "howrah",
        "uttar pradesh",
        "greater noida",
        "noida",
        "bangalore",
        "karnataka",
        "chennai",
        "delhi",
        "new delhi",
        "mumbai",
        "pune",
        "hyderabad",
        "goa",
        "west bengal",
        "tamil nadu",
        "maharashtra",
        "bihar",
        "kerala",
        "andhra pradesh",
        "telangana",
        "uttarakhand",
        "gujarat",
        "punjab",
        "haryana",
        "jammu",
        "kashmir",
        "assam",
        "meghalaya",
        "manipur",
        "mizoram",
        "tripura",
        "sikkim",
        "nagaland",
        "arunachal pradesh",
        "odisha",
        "jharkhand",
        "chhattisgarh",
        "himachal pradesh",
        "rajasthan",
        "madhya pradesh",
        "uttar pradesh",
        "uttaranchal",
        "ahmedabad",
        "surat",
        "indore",
        "nagpur",
        "bhopal",
        "lucknow",
        "patna",
        "jaipur",
        "bhubaneswar",
        "gurgaon",
        "gurugram",
    }

    def __init__(self, service: WeatherService, rag_service: Optional[RAGService] = None) -> None:
        self.service = service
        self.rag_service = rag_service

    @traceable(name="weather_lookup")
    def __call__(self, state: AgentState) -> AgentState:
        candidates, context_docs = self._extract_location_candidates(state)
        last_error: Optional[Exception] = None

        for location in candidates:
            try:
                result = self.service.fetch_weather(location)
            except requests.HTTPError as exc:
                last_error = exc
                status = getattr(exc.response, "status_code", None)
                if status in {400, 404}:
                    LOGGER.warning("Weather lookup failed for %s (status %s)", location, status)
                    continue
                raise
            LOGGER.info("Weather lookup success for %s", result.location)
            return {
                "weather_summary": result.to_summary(),
                "weather_payload": {
                    "location": result.location,
                    "description": result.description,
                    "temperature_c": result.temperature_c,
                    "feels_like_c": result.feels_like_c,
                    "humidity": result.humidity,
                },
                "context_documents": context_docs,
            }

        if last_error is not None:
            raise last_error
        raise ValueError("Unable to determine a valid location for weather lookup")

    def _extract_location_candidates(self, state: AgentState) -> Tuple[List[str], List[str]]:
        question = state["question"]
        question_lower = question.lower()
        tokens = self._tokenize(question)

        scores: Dict[str, int] = {}
        names: Dict[str, str] = {}
        order: Dict[str, int] = {}
        counter = 0
        context_docs: List[str] = []

        def add_candidate(raw: str, weight: int = 0) -> None:
            nonlocal counter
            if not raw:
                return
            candidate = re.sub(r"\s+", " ", raw.strip())
            if not candidate:
                return
            words = candidate.split()
            if len(words) == 0 or len(words) > 4:
                return
            if any(not word.isalpha() for word in words):
                return
            lower_words = [word.lower() for word in words]
            if not (
                " ".join(lower_words) in self.KNOWN_LOCATIONS
                or any(word in self.LOCATION_KEYWORDS for word in lower_words)
            ):
                return
            formatted = " ".join(word.title() for word in lower_words)
            key = formatted.lower()
            if key not in names:
                names[key] = formatted
                order[key] = counter
            elif weight <= scores.get(key, -999):
                counter += 1
                return
            scores[key] = weight
            counter += 1

        needs_context = False
        for idx, token in enumerate(tokens):
            if token in {"in", "at", "from"}:
                tail = tokens[idx + 1 :]
                normalised = self._normalise_tokens(tail)
                if not normalised:
                    continue
                preceding = tokens[max(0, idx - 3) : idx]
                negated = any(t in self.NEGATION_TOKENS for t in preceding)
                weight = 15 if not negated else -5
                add_candidate(" ".join(normalised), weight)
                if negated:
                    needs_context = True

        emphasised_tokens = {"current", "currently", "now", "today"}
        if emphasised_tokens & set(tokens):
            needs_context = True

        for name in self.KNOWN_LOCATIONS:
            if name in question_lower:
                add_candidate(name.title(), 12)

        if self.rag_service is not None and (needs_context or not scores):
            self.rag_service.ensure_ingested()
            documents = self.rag_service.retrieve(question)
            context_docs = [doc.page_content for doc in documents[: self.MAX_DOC_CANDIDATES]]
            for doc in documents:
                for location in self._extract_locations_from_text(doc.page_content):
                    add_candidate(location, 8)
                    if len(scores) >= self.MAX_CANDIDATES:
                        break
                if len(scores) >= self.MAX_CANDIDATES:
                    break

        if not scores:
            add_candidate("London", -10)

        ordered_keys = sorted(
            scores.keys(),
            key=lambda key: (-scores[key], order[key]),
        )[: self.MAX_CANDIDATES]
        ordered = [names[key] for key in ordered_keys]
        return ordered, context_docs

    def _extract_locations_from_text(self, text: str) -> List[str]:
        compact = re.sub(r"\s+", " ", text)
        results: List[str] = []
        patterns = [
            r"from\s+([A-Z][a-zA-Z\s]+)",
            r"based in\s+([A-Z][a-zA-Z\s]+)",
            r"living in\s+([A-Z][a-zA-Z\s]+)",
            r"lives in\s+([A-Z][a-zA-Z\s]+)",
            r"resides in\s+([A-Z][a-zA-Z\s]+)",
            r"located in\s+([A-Z][a-zA-Z\s]+)",
            r"headquartered in\s+([A-Z][a-zA-Z\s]+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                candidate = match.group(1).strip()
                if candidate:
                    results.append(candidate)
        for match in re.finditer(
            r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})\s*,\s*(India|United Kingdom|United States|USA|Canada|Australia|Singapore)",
            compact,
        ):
            results.append(match.group(1))
        lower = compact.lower()
        for name in self.KNOWN_LOCATIONS:
            if name in lower:
                results.append(name.title())
        seen: set[str] = set()
        ordered: List[str] = []
        for candidate in results:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(candidate)
            if len(ordered) >= self.MAX_DOC_CANDIDATES:
                break
        return ordered

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        sanitized = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
        return [token for token in sanitized.split() if token]

    @staticmethod
    def _normalise_tokens(tokens: List[str]) -> List[str]:
        discard_words = {
            "what",
            "whats",
            "is",
            "the",
            "weather",
            "forecast",
            "tell",
            "me",
            "about",
            "his",
            "her",
            "their",
            "location",
            "current",
        }
        stop_words = {"now", "right", "currently", "today", "outside", "please", "thanks", "thank"}
        cleaned: List[str] = []
        for token in tokens:
            if token in stop_words:
                break
            if token in discard_words:
                continue
            cleaned.append(token)
        return cleaned
class RagNode:
    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    @traceable(name="rag_retrieval")
    def __call__(self, state: AgentState) -> AgentState:
        question = state["question"]
        self.rag_service.ensure_ingested()
        documents = self.rag_service.retrieve(question)
        LOGGER.info("RAG retrieved %d chunks", len(documents))
        return {
            "context_documents": [doc.page_content for doc in documents],
            "weather_summary": "",
            "weather_payload": {},
        }


class LLMResponder:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self.model = llm_provider.get_chat_model()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "PROFILE"
                    "Core Capabilities:"
                    "Understand and respond to questions based solely on the knowledge extracted from the provided context."
                    "Present answers in a concise, informative, and professional manner."
                    "TASK INSTRUCTION"
                    "Goal: Help employees find answers to their work-related questions within the company's knowledge base."
                    "Instructions:"
                    "You are an AI named NeuraAI. Introduce yourself briefly."
                    "Do not Introduce on every response. Only introduce once per session."
                    "Restriction: You are only allowed to answer questions that are directly related to the provided context below."
                    "Your answers must be based solely on the provided context section. Avoid introducing external information or making assumptions that are not supported by the context."
                    "Cite weather data explicitly if available and otherwise use document context. If you do not have "
                    "enough information, clearly say so.",
                ),
                (
                    "human",
                    "Question: {question}\n\nWeather: {weather}\n\nContext: {context}\n\nRespond succinctly.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    @traceable(name="llm_response")
    def __call__(self, state: AgentState) -> AgentState:
        weather = state.get("weather_summary", "")
        context = "\n\n".join(state.get("context_documents", [])[:4])
        chain = self.prompt | self.model | self.parser
        answer = chain.invoke({
            "question": state["question"],
            "weather": weather,
            "context": context,
        })
        LOGGER.info("LLM produced answer with %d characters", len(answer))
        return {"answer": answer}


def build_graph(settings: Optional[Settings] = None) -> StateGraph:
    settings = settings or get_settings()
    llm_provider = LLMProvider(settings)
    weather_service = WeatherService(settings.openweather_api_key)
    vector_manager = VectorStoreManager(settings=settings, embeddings=llm_provider.get_embeddings())
    rag_service = RAGService(settings=settings, vector_manager=vector_manager)

    graph = StateGraph(AgentState)
    graph.add_node("router", RouteDecider())
    graph.add_node("weather", WeatherNode(weather_service, rag_service))
    graph.add_node("rag", RagNode(rag_service))
    graph.add_node("llm", LLMResponder(llm_provider))

    graph.set_entry_point("router")

    def choose_route(state: AgentState) -> str:
        route = state.get("route")
        if route not in {"weather", "rag"}:
            raise ValueError(f"Unsupported route: {route}")
        return route

    graph.add_conditional_edges("router", choose_route, {"weather": "weather", "rag": "rag"})
    graph.add_edge("weather", "llm")
    graph.add_edge("rag", "llm")
    graph.add_edge("llm", END)

    return graph


def create_agent(settings: Optional[Settings] = None):
    """Compile and return a runnable LangGraph."""
    return build_graph(settings).compile()


__all__ = ["create_agent", "build_graph", "AgentState"]
