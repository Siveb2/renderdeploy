# app/llm_core.py
import os
import logging
import asyncio
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError, APIError

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for the LLM Provider, loaded from environment variables."""
    api_key: str = field(repr=False)
    model: str
    max_tokens: int
    temperature: float
    timeout: int

    @classmethod
    def from_env(cls):
        """Creates a configuration instance from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        return cls(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1500)),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            timeout=int(os.getenv("LLM_TIMEOUT", 30))
        )

class LLMProvider:
    """Handles communication with the OpenRouter API with exponential backoff."""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    async def generate(self, messages: List[Dict], max_tokens_override: Optional[int] = None) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.model,
                    messages=messages,
                    max_tokens=max_tokens_override or self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                )
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
                raise APIError("Empty response from LLM API", response=None, body=None)
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after multiple retries.")
                    raise e
            except Exception as e:
                logger.error(f"LLM provider error: {e}", exc_info=True)
                raise e
        # This line should not be reached if retries fail, as the exception will be raised.
        # It's here as a fallback.
        raise Exception("LLM generation failed after multiple retries.")


class PersonaManager:
    """Loads and manages the AI's persona."""
    def __init__(self, persona_file_path: str):
        self.persona_content = "You are a helpful AI assistant."
        try:
            with open(persona_file_path, 'r', encoding='utf-8') as f:
                self.persona_content = f.read().strip()
        except Exception:
            logger.warning(f"Persona file not found at '{persona_file_path}'. Using default.")

    def get_system_prompt(self, summary: Optional[str] = None) -> str:
        prompt = f"PERSONA:\n{self.persona_content}"
        if summary:
            prompt += f"\n\nCONTEXT SUMMARY:\n{summary}"
        return prompt

class LLMEngine:
    """Orchestrates all stateless LLM calls."""
    def __init__(self, config: LLMConfig, persona_file_path: str):
        self.llm_provider = LLMProvider(config)
        self.persona_manager = PersonaManager(persona_file_path)

    async def get_response(self, user_input: str, history: List[Dict], summary: Optional[str]) -> str:
        system_prompt = self.persona_manager.get_system_prompt(summary)
        messages = [{'role': 'system', 'content': system_prompt}] + history + [{'role': 'user', 'content': user_input}]
        return await self.llm_provider.generate(messages)

    async def get_summary(self, history_chunk: List[Dict]) -> str:
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_chunk])
        summary_prompt = f"Concisely summarize the key points of the following conversation: {conversation_text}"
        messages = [{'role': 'user', 'content': summary_prompt}]
        return await self.llm_provider.generate(messages, max_tokens_override=500)
    
    async def health_check(self) -> Dict[str, Any]:
        start_time = time.time()
        try:
            await self.llm_provider.generate([{'role': 'user', 'content': 'Ping'}], max_tokens_override=5)
            latency = time.time() - start_time
            return {"healthy": True, "details": "LLM API is responsive", "latency_seconds": round(latency, 4)}
        except Exception as e:
            latency = time.time() - start_time
            return {"healthy": False, "details": str(e), "latency_seconds": round(latency, 4)}
