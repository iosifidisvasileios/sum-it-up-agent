"""
Sum-It-Up Agent - Main orchestrator for audio processing pipeline.

This module provides the main AI agent that:
- Parses user prompts to extract intents and requirements
- Validates input files
- Orchestrates the entire pipeline using MCP servers
- Handles communication channels (email, slack, etc.)
- Provides error handling and logging
"""

from .orchestrator import AudioProcessingAgent
from .prompt_parser import PromptParser, UserIntent, OllamaProvider
from .models import AgentConfig, PipelineResult
from .logger import get_agent_logger

__all__ = [
    "AudioProcessingAgent",
    "OllamaProvider",
    "PromptParser",
    "UserIntent",
    "AgentConfig",
    "PipelineResult",
    "get_agent_logger"
]
