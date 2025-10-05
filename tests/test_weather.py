import pytest
import responses

from ai_pipeline.testing_utils import describe_test
from ai_pipeline.weather import WeatherService


@describe_test(
    purpose="Confirms WeatherService hits OpenWeatherMap and normalises the JSON payload into a WeatherResult instance.",
    targets=[WeatherService, WeatherService.fetch_weather],
)
@responses.activate
def test_weather_service_fetch_weather_success():
    """Simulate a successful API call and ensure the returned WeatherResult preserves key fields."""
    api_key = "test-key"
    service = WeatherService(api_key=api_key, base_url="https://api.openweathermap.org/data/2.5/weather")
    payload = {
        "name": "Paris",
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 19.5, "feels_like": 18.0, "humidity": 50},
    }
    responses.add(
        responses.GET,
        service.base_url,
        json=payload,
        status=200,
        match=[responses.matchers.query_param_matcher({"q": "Paris", "appid": api_key, "units": "metric"})],
    )

    result = service.fetch_weather("Paris")

    assert result.location == "Paris"
    assert "clear sky" in result.description
    assert pytest.approx(result.temperature_c, 0.1) == 19.5
    assert result.humidity == 50
