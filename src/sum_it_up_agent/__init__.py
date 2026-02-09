"""Sum-It-Up Agent: Agentic AI for meeting transcription, summarization, and topic classification."""

__version__ = "0.1.0"

# Main agent exports
from .agent import AudioProcessingAgent, AgentConfig, UserIntent, PipelineResult

__all__ = [
    "AudioProcessingAgent",
    "AgentConfig", 
    "UserIntent",
    "PipelineResult"
]
