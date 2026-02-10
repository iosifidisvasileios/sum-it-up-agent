from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


class SummarizationStatus(Enum):
    """Status of summarization process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TranscriptionSegment:
    """Transcription segment data."""
    start_time: float
    end_time: float
    speaker: str
    text: str
    
    def __post_init__(self):
        self.text = self.text.strip()


@dataclass
class SummarizationRequest:
    """Request for summarization."""
    file_path: str
    meeting_type: str
    segments: List[TranscriptionSegment]
    metadata: Optional[Dict[str, Any]] = None
    user_preferences: Optional[List[str]] = None
    
    def get_transcript_text(self) -> str:
        """Build transcript text from segments."""
        lines = []
        for segment in self.segments:
            lines.append(f"[{segment.start_time:.2f}-{segment.end_time:.2f}] {segment.speaker}: {segment.text}")
        return "\n".join(lines)

    def get_user_preferences(self) -> str:
        """Get user preferences."""
        if not self.user_preferences:
            return ""
        return "### SPECIAL USER INSTRUCTIONS, PAY EXTRA ATTENTION TO THESE INSTRUCTIONS!\n\nUSER:" + "\n- ".join(self.user_preferences)

    def get_time_range(self) -> Dict[str, float]:
        """Get time range of the transcription."""
        if not self.segments:
            return {"start_sec": 0.0, "end_sec": 0.0}
        
        start_times = [s.start_time for s in self.segments]
        end_times = [s.end_time for s in self.segments]
        
        return {
            "start_sec": min(start_times),
            "end_sec": max(end_times)
        }
    
    def get_participants(self) -> List[str]:
        """Get unique participants."""
        participants = list(set(s.speaker for s in self.segments))
        return sorted(participants)


@dataclass
class SummarizationResult:
    """Result of summarization process."""
    request_id: str
    file_path: str
    meeting_type: str
    status: SummarizationStatus
    summary_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    llm_provider: Optional[LLMProvider] = None
    model_name: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    created_at: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if summarization was successful."""
        return self.status == SummarizationStatus.COMPLETED and self.summary_data is not None
    
    def get_summary_json(self) -> Optional[str]:
        """Get summary as JSON string."""
        if self.summary_data:
            return json.dumps(self.summary_data, indent=2, ensure_ascii=False)
        return None
    
    def save_to_file(self, output_path: str) -> None:
        """Save result to file."""
        output_data = {
            "request_id": self.request_id,
            "file_path": self.file_path,
            "meeting_type": self.meeting_type,
            "status": self.status.value,
            "summary_data": self.summary_data,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "llm_provider": self.llm_provider.value if self.llm_provider else None,
            "model_name": self.model_name,
            "token_usage": self.token_usage,
            "created_at": self.created_at
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""
    llm_provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Provider-specific settings
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None
    huggingface_model: Optional[str] = None
    ollama_host: Optional[str] = None
    
    # Processing settings
    batch_size: int = 1
    concurrent_requests: int = 1
    validate_json_output: bool = True
    
    def __post_init__(self):
        # Set default model names based on provider
        if self.model_name == "gpt-4-turbo-preview":
            if self.llm_provider == LLMProvider.OPENAI:
                self.model_name = "gpt-4-turbo-preview"
            elif self.llm_provider == LLMProvider.ANTHROPIC:
                self.model_name = "claude-3-sonnet-20240229"
            elif self.llm_provider == LLMProvider.AZURE_OPENAI:
                self.model_name = self.azure_deployment or "gpt-4"
            elif self.llm_provider == LLMProvider.HUGGINGFACE:
                self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            elif self.llm_provider == LLMProvider.OLLAMA:
                self.model_name = "llama2"


class ISummarizer(ABC):
    """Interface for meeting summarization using LLMs."""
    
    @abstractmethod
    def summarize_transcription(self, request: SummarizationRequest) -> SummarizationResult:
        """Summarize a transcription using appropriate prompt template."""
        pass
    
    @abstractmethod
    def get_supported_meeting_types(self) -> List[str]:
        """Get list of supported meeting types."""
        pass
    
    @abstractmethod
    def validate_request(self, request: SummarizationRequest) -> bool:
        """Validate summarization request."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
