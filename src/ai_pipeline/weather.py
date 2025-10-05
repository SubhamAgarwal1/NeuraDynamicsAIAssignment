"""
Weather API client abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class WeatherResult:
    location: str
    description: str
    temperature_c: float
    feels_like_c: float
    humidity: int

    def to_summary(self) -> str:
        """Return a concise natural-language summary of the weather observation."""
        return (
            f"Weather in {self.location}: {self.description}. "
            f"Temperature {self.temperature_c:.1f} C (feels like {self.feels_like_c:.1f} C), "
            f"humidity {self.humidity}%"
        )


class WeatherService:
    """Handles interaction with the OpenWeatherMap REST API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5/weather", session: Optional[requests.Session] = None) -> None:
        """Initialise the service with credentials, base endpoint and optional requests session."""
        if not api_key:
            raise ValueError("OpenWeatherMap API key is required")
        self.api_key = api_key
        self.base_url = base_url
        self.session = session or requests.Session()

    def fetch_weather(self, location: str, units: str = "metric") -> WeatherResult:
        """Call the OpenWeatherMap API and map the JSON payload into a WeatherResult dataclass."""
        params = {"q": location, "appid": self.api_key, "units": units}
        LOGGER.info("Fetching weather for %s", location)
        response = self.session.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        main = payload.get("main", {})
        weather = payload.get("weather", [{}])[0]
        return WeatherResult(
            location=payload.get("name", location),
            description=weather.get("description", "No description"),
            temperature_c=float(main.get("temp", 0.0)),
            feels_like_c=float(main.get("feels_like", 0.0)),
            humidity=int(main.get("humidity", 0)),
        )


__all__ = ["WeatherService", "WeatherResult"]
