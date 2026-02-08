"""
Topic Classification Package

A production-grade zero-shot topic classification library for conversations.
"""

from .interfaces import (
    ITopicClassifier,
    TopicClassificationConfig,
    ClassificationResult,
    ConversationFormat,
    EnsembleMethod,
    DeviceType
)
from .topic_classifier import TopicClassifier
from .factory import TopicClassifierFactory, ClassifierType
from .use_cases import TopicClassificationUseCase

__version__ = "1.0.0"
__all__ = [
    "ITopicClassifier",
    "TopicClassificationConfig",
    "ClassificationResult",
    "ConversationFormat",
    "EnsembleMethod",
    "DeviceType",
    "TopicClassifier",
    "TopicClassifierFactory",
    "ClassifierType",
    "TopicClassificationUseCase"
]
