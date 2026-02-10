"""
Prompt parser for extracting user intents and requirements using multiple LLM providers.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .models import UserIntent, CommunicationChannel, SummaryType
from .logger import get_agent_logger


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def extract_intent(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Extract intent from prompt using the LLM."""
        pass



class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, model: str = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.logger = get_agent_logger("ollama_provider")
        # Ollama doesn't require API key for local use
    
    async def extract_intent(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Extract intent using Ollama."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        
        self.logger.info(f"Extracting intent using Ollama model: {self.model}")
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{system_prompt}\n\nExtract intent from this prompt: {prompt}",
                "stream": False,
                "keep_alive": 0,   # <-- unload immediately after response
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000
                }
            }
        )
        
        if response.status_code != 200:
            self.logger.error(f"Ollama request failed with status {response.status_code}: {response.text}")
            raise RuntimeError(f"Ollama request failed: {response.text}")
        
        content = response.json()["response"].strip()
        self.logger.debug(f"Ollama response: {content}")
        
        # Extract JSON from response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_content = content[json_start:json_end].strip()
        else:
            json_content = content
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from Ollama response: {e}")
            raise


class PromptParser:
    """Parses user prompts to extract intent using configurable LLM providers."""
    
    def __init__(
        self, 
        provider: str = "ollama",
        model: Optional[str] = "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
        base_url: Optional[str] = "http://localhost:11434",
        prompt_limit: int = 256
    ):
        """
        Initialize the LLM-based prompt parser.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "ollama")
            model: Model name (provider-specific default if None)
            base_url: Base URL for Ollama or custom endpoints
        """
        self.provider = self._create_provider(provider, model, base_url)
        self.prompt_limit = prompt_limit
        self.logger = get_agent_logger("prompt_parser")

    @staticmethod
    def _create_provider(
        provider: str,
        model: Optional[str], 
        base_url: Optional[str]
    ) -> LLMProvider:
        """Create the appropriate LLM provider."""
        
        if provider.lower() == "ollama":
            model = model or "llama3.1"
            base_url = base_url or "http://localhost:11434"
            return OllamaProvider(model=model, base_url=base_url)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")
    
    async def parse_prompt(self, prompt: str) -> UserIntent:
        """
        Parse user prompt and extract intent using configured LLM.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            UserIntent object with extracted requirements
        """
        if len(prompt) > self.prompt_limit:
            self.logger.warning(f"Prompt longer than {self.prompt_limit} chars, skipping")
            return self._create_fallback_intent(prompt)

        try:
            intent_json = await self._extract_intent_with_llm(prompt)
            return self._json_to_intent(intent_json)
        except Exception as e:
            self.logger.error(f"Failed to parse prompt: {e}")
            return self._create_fallback_intent(prompt)
    
    async def _extract_intent_with_llm(self, prompt: str) -> dict:
        """
        Extract intent from prompt using configured LLM.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Dictionary containing extracted intent
        """
        system_prompt = """You are an expert at analyzing user requests for audio processing and summarization tasks. 
Extract the user's intent and return a structured JSON response.

Analyze the user's prompt for:
1. Communication channels (email, slack, discord, telegram, jira)
2. Summary types (action_items, decisions, key_points, detailed, bullet_points, executive, standard)
3. Recipients (email addresses or names)
4. Custom instructions (tone, voice/persona, audience, depth, don’ts)

Return a JSON object with this exact structure:
{
    "wants_summary": True/False,
    "wants_transcription": True/False,
    "communication_channels": ["email", "slack", "jira", etc.],
    "recipients": ["email@example.com", "Bill_Slack"],
    "subject": "Email subject or null",
    "summary_types": ["action_items", "decisions", etc.],
    "custom_instructions": ["dont use emojis", "be direct", "keep it professional"]
}

Be precise and only include values that are clearly indicated in the prompt. 
If something is not mentioned, use null or empty arrays. 
Default wants_summary to true unless user explicitly says "transcription only" or "no summary".
Default wants_transcription to False unless user explicitly says "transcription" or "transcribe" or "no summary"."""

        return await self.provider.extract_intent(prompt, system_prompt)

    @staticmethod
    def _json_to_intent(intent_data: dict) -> UserIntent:
        """
        Convert LLM JSON response to UserIntent object.
        
        Args:
            intent_data: Dictionary from LLM response
            
        Returns:
            UserIntent object
        """
        intent = UserIntent()
        
        # Basic requirements
        intent.wants_summary = intent_data.get("wants_summary", True)
        intent.wants_transcription = intent_data.get("wants_transcription", False)
        
        # Communication channels
        channels_str = intent_data.get("communication_channels", [])
        intent.communication_channels = []
        for channel_str in channels_str:
            try:
                intent.communication_channels.append(CommunicationChannel(channel_str))
            except ValueError:
                # Skip invalid channels
                pass
        
        # Recipients
        intent.recipients = intent_data.get("recipients", [])
        
        # Subject
        intent.subject = intent_data.get("subject")
        
        # Summary types
        summary_types_str = intent_data.get("summary_types", [])
        intent.summary_types = []
        for summary_type_str in summary_types_str:
            try:
                intent.summary_types.append(SummaryType(summary_type_str))
            except ValueError:
                # Skip invalid summary types
                pass
        
        # Default to standard if no specific types requested
        if not intent.summary_types and intent.wants_summary:
            intent.summary_types.append(SummaryType.STANDARD)
        
        # Custom instructions
        intent.custom_instructions = intent_data.get("custom_instructions", [])
        
        return intent

    @staticmethod
    def _create_fallback_intent(prompt: str) -> UserIntent:
        """
        Create a basic fallback intent when LLM fails.
        
        Args:
            prompt: Original user prompt
            error: Error message from LLM failure
            
        Returns:
            Basic UserIntent object
        """
        intent = UserIntent()
        
        # Basic keyword-based fallback for critical functionality
        prompt_lower = prompt.lower()
        
        # Check for transcription only
        if any(word in prompt_lower for word in ['transcription only', 'just transcribe', 'no summary']):
            intent.wants_summary = False
            intent.wants_transcription = True
        else:
            intent.wants_summary = True
            intent.wants_transcription = False
        
        # Basic email detection
        if 'email' in prompt_lower or 'mail' in prompt_lower:
            intent.communication_channels.append(CommunicationChannel.EMAIL)
        
        # Basic summary type detection
        if 'action' in prompt_lower and 'item' in prompt_lower:
            intent.summary_types.append(SummaryType.ACTION_ITEMS)
        elif 'decision' in prompt_lower:
            intent.summary_types.append(SummaryType.DECISIONS)
        elif 'detailed' in prompt_lower:
            intent.summary_types.append(SummaryType.DETAILED)
        else:
            intent.summary_types.append(SummaryType.STANDARD)
        
        return intent
