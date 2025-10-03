import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


from ai_pipeline.config import get_settings
settings = get_settings()
print("Key loaded:", bool(settings.openweather_api_key))
print("Prefix:", settings.openweather_api_key[:5])