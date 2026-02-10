import logging
from typing import Optional, Dict, Any
from enum import Enum

from .interfaces import IAudioProcessor, AudioProcessingConfig, DeviceType
from .audio_processor import AudioProcessor


class ProcessorType(Enum):
    STANDARD = "standard"
    FAST = "fast"
    HIGH_QUALITY = "high_quality"


class AudioProcessorFactory:
    """Factory for creating audio processor instances with different configurations."""
    
    _PRESET_CONFIGS = {
        ProcessorType.STANDARD: AudioProcessingConfig(
            device=DeviceType.CUDA,
            whisper_model="base",
            compute_type="float16",
            merge_gap=0.6,
            vad_filter=True
        ),
        ProcessorType.FAST: AudioProcessingConfig(
            device=DeviceType.CUDA,
            whisper_model="tiny",
            compute_type="int8",
            merge_gap=1.0,
            vad_filter=False
        ),
        ProcessorType.HIGH_QUALITY: AudioProcessingConfig(
            device=DeviceType.CUDA,
            whisper_model="large-v3",
            compute_type="float16",
            merge_gap=0.3,
            vad_filter=True
        )
    }
    
    @classmethod
    def create_processor(
        cls,
        processor_type: ProcessorType = ProcessorType.STANDARD,
        huggingface_token: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> IAudioProcessor:
        """Create an audio processor with preset or custom configuration."""
        logger = logger or logging.getLogger(__name__)
        
        # Get base configuration
        config = cls._PRESET_CONFIGS[processor_type]
        
        # Create a copy to avoid modifying the preset
        config_dict = {
            'device': config.device,
            'sample_rate': config.sample_rate,
            'whisper_model': config.whisper_model,
            'compute_type': config.compute_type,
            'diarization_model': config.diarization_model,
            'huggingface_token': huggingface_token,
            'merge_gap': config.merge_gap,
            'vad_filter': config.vad_filter
        }
        
        # Apply overrides
        if config_overrides:
            config_dict.update(config_overrides)
        
        # Create final config
        final_config = AudioProcessingConfig(**config_dict)
        
        # Validate required token
        if not final_config.huggingface_token:
            raise ValueError("HuggingFace token is required for diarization")
        
        logger.info(f"Creating {processor_type.value} audio processor")
        return AudioProcessor(final_config, logger=logger)
    
    @classmethod
    def create_custom_processor(cls, config: AudioProcessingConfig, logger: Optional[logging.Logger] = None) -> IAudioProcessor:
        """Create an audio processor with custom configuration."""
        logger = logger or logging.getLogger(__name__)
        
        if not config.huggingface_token:
            raise ValueError("HuggingFace token is required for diarization")
        
        logger.info("Creating custom audio processor")
        return AudioProcessor(config, logger=logger)
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, ProcessorType]:
        """Get all available processor presets."""
        return {preset.value: preset for preset in ProcessorType}
    
    @classmethod
    def get_preset_config(cls, processor_type: ProcessorType) -> AudioProcessingConfig:
        """Get the configuration for a specific preset (without token)."""
        return cls._PRESET_CONFIGS[processor_type]
