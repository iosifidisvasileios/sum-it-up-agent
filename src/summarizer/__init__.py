"""
Summarizer Package

A production-grade meeting summarization library using LLMs and prompt templates.
"""

from .interfaces import (
    ISummarizer,
    SummarizationConfig,
    SummarizationRequest,
    SummarizationResult,
    SummarizationStatus,
    LLMProvider,
    TranscriptionSegment
)
from .summarizer import Summarizer
from .factory import SummarizerFactory, SummarizerType
from .use_cases import SummarizationUseCase

__version__ = "1.0.0"
__all__ = [
    "ISummarizer",
    "SummarizationConfig",
    "SummarizationRequest",
    "SummarizationResult",
    "SummarizationStatus",
    "LLMProvider",
    "TranscriptionSegment",
    "Summarizer",
    "SummarizerFactory",
    "SummarizerType",
    "SummarizationUseCase"
]
