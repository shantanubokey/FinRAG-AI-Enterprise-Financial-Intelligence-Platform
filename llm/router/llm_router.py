"""
LLM Router — selects the right model based on query complexity and cost.
Supports OpenAI, Ollama (local), and Gemini with a unified interface.
"""

from typing import Any

from config.logging_config import get_logger
from config.settings import AppSettings

logger = get_logger(__name__)


class LLMRouter:
    """
    Routes to the configured LLM provider.
    All providers return the same response dict format.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.provider = settings.default_llm_provider

    async def agenerate(self, prompt: str | list[dict]) -> dict[str, Any]:
        """
        Generate a response. Returns:
        {
          "content": str,
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "cost_usd": float,
        }
        """
        if self.provider == "openai":
            return await self._openai(prompt)
        elif self.provider == "ollama":
            return await self._ollama(prompt)
        elif self.provider == "gemini":
            return await self._gemini(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def _openai(self, prompt: str | list[dict]) -> dict[str, Any]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt
        )

        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=self.settings.openai_temperature,
        )

        usage = response.usage
        # GPT-4o-mini pricing (per 1K tokens)
        cost = (usage.prompt_tokens * 0.00015 + usage.completion_tokens * 0.0006) / 1000

        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(cost, 6),
        }

    async def _ollama(self, prompt: str | list[dict]) -> dict[str, Any]:
        import httpx

        content = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={"model": self.settings.ollama_model, "prompt": content, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

        return {
            "content": data.get("response", ""),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            "cost_usd": 0.0,  # Local model, no cost
        }

    async def _gemini(self, prompt: str | list[dict]) -> dict[str, Any]:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        content = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        response = await model.generate_content_async(content)

        return {
            "content": response.text,
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
            "cost_usd": 0.0,
        }
