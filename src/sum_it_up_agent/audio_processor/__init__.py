"""
Audio Processor Package

A production-grade audio processing library with speaker diarization and transcription capabilities.
"""

from .interfaces import (
    IAudioProcessor,
    AudioProcessingConfig,
    TranscriptionSegment,
    DeviceType
)
from .audio_processor import AudioProcessor
from .factory import AudioProcessorFactory, ProcessorType
from .use_cases import AudioProcessingUseCase

__version__ = "1.0.0"
__all__ = [
    "IAudioProcessor",
    "AudioProcessingConfig",
    "TranscriptionSegment",
    "DeviceType",
    "AudioProcessor",
    "AudioProcessorFactory",
    "ProcessorType",
    "AudioProcessingUseCase",
]
