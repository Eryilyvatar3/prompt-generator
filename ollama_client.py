"""
Клиент для работы с локальным Ollama API.
"""

import requests
from typing import Optional


class OllamaClient:
    """Обёртка над HTTP API Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Проверяет, доступен ли Ollama и загружена ли модель."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            return self.model in models
        except requests.RequestException:
            return False

    def generate(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """
        Отправляет промпт в модель и возвращает ответ.

        Args:
            prompt: текст промпта.
            temperature: креативность модели (0.0 — детерминированно).

        Returns:
            str или None при ошибке.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=600,  # 10 минут — LLM на CPU может думать долго
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as e:
            print(f"[OllamaClient] Ошибка запроса: {e}")
            return None