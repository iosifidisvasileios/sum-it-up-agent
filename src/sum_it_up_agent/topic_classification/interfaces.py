from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class ConversationFormat(Enum):
    ONE_ON_ONE = "one_on_one"
    GROUP_MEETING = "group_meeting"
    UNKNOWN = "unknown"


class EnsembleMethod(Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAX = "max"
    VOTING = "voting"


@dataclass
class ClassificationResult:
    """Result of topic classification."""
    file_path: str
    conversation_format: ConversationFormat
    predicted_topic: str
    confidence: float
    ensemble_scores: Dict[str, float]
    per_model_scores: Dict[str, Dict[str, float]]
    num_segments_used: int
    num_segments_total: int
    text_length: int
    processing_time: Optional[float] = None


@dataclass
class TopicClassificationConfig:
    """Configuration for topic classification."""
    device: DeviceType = DeviceType.CPU
    models: List[str] = None
    labels: List[str] = None
    hypothesis_template: str = "This conversation is about {}."
    multi_label: bool = False
    truncation: bool = True
    ensemble_method: EnsembleMethod = EnsembleMethod.MEAN
    first_n_segments: Optional[int] = None
    min_text_length: int = 50
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                "FacebookAI/roberta-large-mnli",
                "cross-encoder/nli-deberta-v3-small",
                "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
            ]
        
        if self.labels is None:
            self.labels = [
                "team status sync / standup",
                "planning / coordination meeting",
                "decision-making meeting",
                "brainstorming session",
                "retrospective / postmortem",
                "training / onboarding",
                "interview",
                "customer call / sales demo",
                "support / incident call",
                "other",
            ]


class ITopicClassifier(ABC):
    """Interface for topic classification operations."""
    
    @abstractmethod
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load and parse JSON data from file."""
        pass
    
    @abstractmethod
    def extract_segments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation segments from JSON data."""
        pass
    
    @abstractmethod
    def determine_conversation_format(self, segments: List[Dict[str, Any]]) -> ConversationFormat:
        """Determine if conversation is one-on-one or group meeting."""
        pass
    
    @abstractmethod
    def build_conversation_text(self, segments: List[Dict[str, Any]]) -> str:
        """Build formatted text from conversation segments."""
        pass
    
    @abstractmethod
    def classify_text(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Classify text using zero-shot classification."""
        pass
    
    @abstractmethod
    def ensemble_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions from multiple models."""
        pass
    
    @abstractmethod
    def classify_conversation(self, file_path: str) -> ClassificationResult:
        """Complete classification pipeline for a conversation file."""
        pass
    
    @abstractmethod
    def batch_classify(self, file_paths: List[str]) -> List[ClassificationResult]:
        """Classify multiple conversation files."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
