from .base import BaseClient
import requests
import os
import logging

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):
    def __init__(self, model: str, api_key: str = None, temperature: float = 1.0):
        super().__init__(model, temperature)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.fatal("Gemini API key required via argument or GEMINI_API_KEY env variable")
            exit(-1)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        # Gemini expects a different message format; adapt as needed
        prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "candidateCount": n}
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        # Adapt return value to match expected format
        return [{"message": {"content": c["content"]["parts"][0]["text"]}} for c in candidates]