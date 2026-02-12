"""
Data models for the AI agent.
"""

from dataclasses import dataclass, field
from enum import Enum
import dotenv
from typing import Any, Dict, List, Optional, Union
import os

dotenv.load_dotenv()

class CommunicationChannel(Enum):
    """Supported communication channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    PDF = "pdf"
    JIRA = "jira"


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    STANDARD = "standard"
    ACTION_ITEMS = "action_items"
    DECISIONS = "decisions"
    KEY_POINTS = "key_points"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"


@dataclass
class UserIntent:
    """Parsed user intent from the prompt."""
    # Core requirements
    wants_summary: bool = True
    wants_transcription: bool = False

    # Communication preferences
    communication_channels: List[CommunicationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    subject: Optional[str] = None

    # Summary customization
    summary_types: List[SummaryType] = field(default_factory=list)

    meeting_type: str = "team status sync / standup"
    custom_instructions: List = None


@dataclass
class AgentConfig:
    """Configuration for the audio processing agent."""
    # File system settings
    allowed_audio_formats: List[str] = field(default_factory=lambda: [
        ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"
    ])
    max_file_size_mb: int = 1_024
    prompt_limit: int = 256
    temp_dir: Optional[str] = None
    
    # MCP server settings
    audio_processor_mcp_url: str = f"http://{os.getenv('MCP_HOST_AUDIO_PROCESSOR')}:{os.getenv('MCP_PORT_AUDIO_PROCESSOR')}{os.getenv('MCP_PATH_AUDIO_PROCESSOR')}"
    topic_classifier_mcp_url: str = f"http://{os.getenv('MCP_HOST_TOPIC_CLASSIFIER')}:{os.getenv('MCP_PORT_TOPIC_CLASSIFIER')}{os.getenv('MCP_PATH_TOPIC_CLASSIFIER')}"
    summarizer_mcp_url: str = f"http://{os.getenv('MCP_HOST_SUMMARIZER')}:{os.getenv('MCP_PORT_SUMMARIZER')}{os.getenv('MCP_PATH_SUMMARIZER')}"
    communicator_mcp_url: str = f"http://{os.getenv('MCP_HOST_COMMUNICATOR')}:{os.getenv('MCP_PORT_COMMUNICATOR')}{os.getenv('MCP_PATH_COMMUNICATOR')}"
    
    # Processing settings
    preset_audio: str = "high_quality"
    preset_topic_classifier: str = "high_accuracy"
    preset_summarizer: str = "ollama_local"
    output_format: str = "json"
    output_dir: str = "/home/vios/PycharmProjects/sum-it-up-agent/examples/ollama_summaries/"

    # Communication settings
    default_email_subject: str = "team status sync / standup"
    max_recipients: int = 50
    
    # AI settings for prompt parsing
    llm_provider: str = "ollama"  # openai, anthropic, ollama
    llm_model: str = "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL"  # provider-specific model
    llm_base_url: Optional[str] = "http://localhost:11434"
    confidence_threshold: float = 0.7


@dataclass
class PipelineStep:
    """Represents a step in the processing pipeline."""
    name: str
    status: str  # pending, running, completed, failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of the entire pipeline execution."""
    success: bool
    user_intent: UserIntent
    input_file: str
    
    # Step results
    audio_processing: Optional[PipelineStep] = None
    topic_classification: Optional[PipelineStep] = None
    summarization: Optional[PipelineStep] = None
    communication: Optional[PipelineStep] = None
    
    # Output files
    transcription_file: Optional[str] = None
    summary_file: Optional[str] = None
    communication_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_duration: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileValidationResult:
    """Result of file validation."""
    is_valid: bool
    file_path: str
    file_size_mb: float
    file_format: str
    exists: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
