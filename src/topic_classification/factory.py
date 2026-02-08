import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from .interfaces import ITopicClassifier, TopicClassificationConfig, DeviceType, EnsembleMethod
from .topic_classifier import TopicClassifier


class ClassifierType(Enum):
    """Predefined classifier configurations."""
    FAST = "fast"
    STANDARD = "standard"
    HIGH_ACCURACY = "high_accuracy"
    LIGHTWEIGHT = "lightweight"


class TopicClassifierFactory:
    """Factory for creating topic classifier instances with different configurations."""
    
    _PRESET_CONFIGS = {
        ClassifierType.FAST: TopicClassificationConfig(
            device=DeviceType.CPU,
            models=["cross-encoder/nli-deberta-v3-small"],
            ensemble_method=EnsembleMethod.MEAN,
            first_n_segments=5,
        ),
        ClassifierType.STANDARD: TopicClassificationConfig(
            device=DeviceType.CUDA,
            models=[
                "FacebookAI/roberta-large-mnli",
                "cross-encoder/nli-deberta-v3-small"
            ],
            ensemble_method=EnsembleMethod.MEAN,
            first_n_segments=10,
        ),
        ClassifierType.HIGH_ACCURACY: TopicClassificationConfig(
            device=DeviceType.CUDA,
            models=[
                "FacebookAI/roberta-large-mnli",
                "cross-encoder/nli-deberta-v3-small",
                "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
            ],
            ensemble_method=EnsembleMethod.MEAN,
            first_n_segments=None,  # Use all segments
        ),
        ClassifierType.LIGHTWEIGHT: TopicClassificationConfig(
            device=DeviceType.CPU,
            models=["MoritzLaurer/deberta-v3-large-zeroshot-v2.0"],
            ensemble_method=EnsembleMethod.MAX,
            first_n_segments=3,
        )
    }
    
    @classmethod
    def create_classifier(
        cls,
        classifier_type: ClassifierType = ClassifierType.STANDARD,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> ITopicClassifier:
        """Create a topic classifier with preset or custom configuration."""
        logger = logging.getLogger(__name__)
        
        # Get base configuration
        config = cls._PRESET_CONFIGS[classifier_type]
        
        # Create a copy to avoid modifying the preset
        config_dict = {
            'device': config.device,
            'models': config.models.copy(),
            'labels': config.labels.copy(),
            'hypothesis_template': config.hypothesis_template,
            'multi_label': config.multi_label,
            'truncation': config.truncation,
            'ensemble_method': config.ensemble_method,
            'first_n_segments': config.first_n_segments,
            'min_text_length': config.min_text_length
        }
        
        # Apply overrides
        if config_overrides:
            config_dict.update(config_overrides)
        
        # Create final config
        final_config = TopicClassificationConfig(**config_dict)
        
        logger.info(f"Creating {classifier_type.value} topic classifier")
        return TopicClassifier(final_config)
    
    @classmethod
    def create_custom_classifier(cls, config: TopicClassificationConfig) -> ITopicClassifier:
        """Create a topic classifier with custom configuration."""
        logger = logging.getLogger(__name__)
        
        logger.info("Creating custom topic classifier")
        return TopicClassifier(config)
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, ClassifierType]:
        """Get all available classifier presets."""
        return {preset.value: preset for preset in ClassifierType}
    
    @classmethod
    def get_preset_config(cls, classifier_type: ClassifierType) -> TopicClassificationConfig:
        """Get the configuration for a specific preset."""
        return cls._PRESET_CONFIGS[classifier_type]
    
    @classmethod
    def create_for_device(
        cls,
        device: DeviceType = DeviceType.CPU,
        performance_level: str = "standard"
    ) -> ITopicClassifier:
        """Create classifier optimized for specific device."""
        
        if device == DeviceType.CPU:
            if performance_level == "fast":
                return cls.create_classifier(ClassifierType.LIGHTWEIGHT)
            else:
                return cls.create_classifier(ClassifierType.FAST)
        else:
            # GPU available
            if performance_level == "fast":
                return cls.create_classifier(ClassifierType.STANDARD)
            elif performance_level == "high_accuracy":
                return cls.create_classifier(ClassifierType.HIGH_ACCURACY)
            else:
                return cls.create_classifier(ClassifierType.STANDARD)
    
    @classmethod
    def create_for_batch_processing(
        cls,
        batch_size: int = 10,
        device: DeviceType = DeviceType.CUDA
    ) -> ITopicClassifier:
        """Create classifier optimized for batch processing."""
        
        if batch_size > 50:
            # Large batch - use lightweight config
            config_overrides = {
                'first_n_segments': 3,
                'device': device
            }
            return cls.create_classifier(ClassifierType.LIGHTWEIGHT, config_overrides)
        elif batch_size > 10:
            # Medium batch - use standard config
            config_overrides = {
                'first_n_segments': 5,
                'device': device
            }
            return cls.create_classifier(ClassifierType.STANDARD, config_overrides)
        else:
            # Small batch - use high accuracy
            config_overrides = {
                'device': device
            }
            return cls.create_classifier(ClassifierType.HIGH_ACCURACY, config_overrides)
    
    @staticmethod
    def _estimate_memory_usage(config: TopicClassificationConfig) -> str:
        """Estimate memory usage for a configuration."""
        if config.device == DeviceType.CPU:
            return "Low"
        
        # Rough estimates based on model count and size
        model_count = len(config.models)
        if model_count >= 3:
            return "High"
        elif model_count >= 2:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def _estimate_processing_speed(config: TopicClassificationConfig) -> str:
        """Estimate processing speed for a configuration."""
        model_count = len(config.models)

        # Simple heuristic based on model count and max length
        if model_count >= 3:
            return "Slow"
        elif model_count >= 2:
            return "Medium"
        else:
            return "Fast"
