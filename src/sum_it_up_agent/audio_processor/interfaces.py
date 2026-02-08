from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class TranscriptionSegment:
    start_time: float
    end_time: float
    speaker: str
    text: str
    
    def __post_init__(self):
        self.text = self.text.strip()


@dataclass
class AudioProcessingConfig:
    device: DeviceType = DeviceType.CUDA
    sample_rate: int = 16000
    whisper_model: str = "large-v3"
    compute_type: str = "float16"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    huggingface_token: Optional[str] = None
    merge_gap: float = 0.6
    vad_filter: bool = True


class IAudioProcessor(ABC):
    """Interface for audio processing operations."""
    
    @abstractmethod
    def convert_audio_format(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Convert audio to WAV format with specified sample rate."""
        pass
    
    @abstractmethod
    def perform_diarization(self, audio_path: str) -> object:
        """Perform speaker diarization on audio file."""
        pass
    
    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> List[Tuple]:
        """Transcribe audio using Whisper model."""
        pass
    
    @abstractmethod
    def merge_segments_by_speaker(self, segments: List[Tuple]) -> List[TranscriptionSegment]:
        """Merge consecutive segments from same speaker."""
        pass
    
    @abstractmethod
    def process_audio(self, audio_path: str) -> List[TranscriptionSegment]:
        """Complete audio processing pipeline: diarization + transcription."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
