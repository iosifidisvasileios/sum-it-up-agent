import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import os

from src.summarizer import ISummarizer, SummarizationConfig, LLMProvider


class SummarizerType(Enum):
    """Predefined summarizer configurations."""
    OPENAI_FAST = "openai_fast"
    OPENAI_STANDARD = "openai_standard"
    OPENAI_HIGH_QUALITY = "openai_high_quality"
    ANTHROPIC_STANDARD = "anthropic_standard"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE_LOCAL = "huggingface_local"
    OLLAMA_LOCAL = "ollama_local"
    COST_OPTIMIZED = "cost_optimized"


class SummarizerFactory:
    """Factory for creating summarizer instances with different configurations."""
    
    _PRESET_CONFIGS = {
        SummarizerType.OPENAI_FAST: SummarizationConfig(
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=8192,
            timeout=120,
            max_retries=2,
            concurrent_requests=3
        ),
        SummarizerType.OPENAI_STANDARD: SummarizationConfig(
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=8192,
            timeout=300,
            max_retries=3,
            concurrent_requests=2
        ),
        SummarizerType.OPENAI_HIGH_QUALITY: SummarizationConfig(
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            temperature=0.05,
            max_tokens=8192,
            timeout=600,
            max_retries=3,
            concurrent_requests=1
        ),
        SummarizerType.ANTHROPIC_STANDARD: SummarizationConfig(
            llm_provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.1,
            max_tokens=8192,
            timeout=300,
            max_retries=3,
            concurrent_requests=2
        ),
        SummarizerType.AZURE_OPENAI: SummarizationConfig(
            llm_provider=LLMProvider.AZURE_OPENAI,
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=8192,
            timeout=300,
            max_retries=3,
            concurrent_requests=2,
            azure_api_version="2024-02-15-preview"
        ),
        SummarizerType.HUGGINGFACE_LOCAL: SummarizationConfig(
            llm_provider=LLMProvider.HUGGINGFACE,
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.1,
            max_tokens=8192,
            timeout=600,
            max_retries=1,
            concurrent_requests=1
        ),
        SummarizerType.OLLAMA_LOCAL: SummarizationConfig(
            llm_provider=LLMProvider.OLLAMA,
            model_name="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
            temperature=0.1,
            max_tokens=8192,
            timeout=600,
            max_retries=1,
            concurrent_requests=1,
            ollama_host="http://localhost:11434"
        ),
        SummarizerType.COST_OPTIMIZED: SummarizationConfig(
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
            max_retries=2,
            concurrent_requests=5
        )
    }
    
    @classmethod
    def create_summarizer(
        cls,
        summarizer_type: SummarizerType = SummarizerType.OPENAI_STANDARD,
        config_overrides: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> ISummarizer:
        """Create a summarizer with preset or custom configuration."""
        logger = logging.getLogger(__name__)
        
        # Get base configuration
        config = cls._PRESET_CONFIGS[summarizer_type]
        
        # Create a copy to avoid modifying the preset
        config_dict = {
            'llm_provider': config.llm_provider,
            'model_name': config.model_name,
            'api_key': api_key or config.api_key,
            'api_base': config.api_base,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'timeout': config.timeout,
            'max_retries': config.max_retries,
            'retry_delay': config.retry_delay,
            'azure_deployment': config.azure_deployment,
            'azure_api_version': config.azure_api_version,
            'huggingface_model': config.huggingface_model,
            'ollama_host': config.ollama_host,
            'batch_size': config.batch_size,
            'concurrent_requests': config.concurrent_requests,
            'validate_json_output': config.validate_json_output
        }
        
        # Apply overrides
        if config_overrides:
            config_dict.update(config_overrides)
        
        # Set API key from environment if not provided
        if not config_dict['api_key']:
            config_dict['api_key'] = cls._get_api_key_from_env(config_dict['llm_provider'])
        
        # Create final config
        final_config = SummarizationConfig(**config_dict)
        
        logger.info(f"Creating {summarizer_type.value} summarizer")
        return cls._create_summarizer_from_config(final_config)
    
    @classmethod
    def create_custom_summarizer(cls, config: SummarizationConfig) -> ISummarizer:
        """Create a summarizer with custom configuration."""
        logger = logging.getLogger(__name__)
        
        # Set API key from environment if not provided
        if not config.api_key:
            config.api_key = cls._get_api_key_from_env(config.llm_provider)
        
        logger.info("Creating custom summarizer")
        return cls._create_summarizer_from_config(config)
    
    @classmethod
    def create_from_environment(cls) -> ISummarizer:
        """Create summarizer from environment variables."""
        logger = logging.getLogger(__name__)
        
        # Get provider from environment
        provider_str = os.getenv("SUMMARIZER_PROVIDER", "openai").lower()
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "azure_openai": LLMProvider.AZURE_OPENAI,
            "huggingface": LLMProvider.HUGGINGFACE,
            "ollama": LLMProvider.OLLAMA
        }
        
        provider = provider_map.get(provider_str, LLMProvider.OPENAI)
        
        # Get model from environment
        model_name = os.getenv("SUMMARIZER_MODEL")
        
        # Create config from environment
        config = SummarizationConfig(
            llm_provider=provider,
            model_name=model_name or cls._get_default_model(provider),
            api_key=os.getenv("SUMMARIZER_API_KEY"),
            api_base=os.getenv("SUMMARIZER_API_BASE"),
            temperature=float(os.getenv("SUMMARIZER_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("SUMMARIZER_MAX_TOKENS", "4000")) if os.getenv("SUMMARIZER_MAX_TOKENS") else None,
            timeout=int(os.getenv("SUMMARIZER_TIMEOUT", "300")),
            max_retries=int(os.getenv("SUMMARIZER_MAX_RETRIES", "3")),
            concurrent_requests=int(os.getenv("SUMMARIZER_CONCURRENT_REQUESTS", "2")),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            azure_api_version=os.getenv("AZURE_API_VERSION"),
            huggingface_model=os.getenv("HUGGINGFACE_MODEL"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        
        logger.info(f"Creating summarizer from environment: {provider.value}")
        return cls._create_summarizer_from_config(config)
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, SummarizerType]:
        """Get all available summarizer presets."""
        return {preset.value: preset for preset in SummarizerType}
    
    @classmethod
    def get_preset_config(cls, summarizer_type: SummarizerType) -> SummarizationConfig:
        """Get the configuration for a specific preset."""
        return cls._PRESET_CONFIGS[summarizer_type]

    
    @classmethod
    def create_for_batch_processing(
        cls,
        batch_size: int = 10,
        provider: Optional[LLMProvider] = None,
        cost_sensitive: bool = False
    ) -> ISummarizer:
        """Create summarizer optimized for batch processing."""
        
        if cost_sensitive:
            # Use cost-optimized configuration
            config_overrides = {
                'concurrent_requests': min(5, batch_size),
                'timeout': 120,
                'max_tokens': 2000
            }
            return cls.create_summarizer(SummarizerType.COST_OPTIMIZED, config_overrides)
        
        if batch_size > 20:
            # Large batch - use fast config with high concurrency
            config_overrides = {
                'concurrent_requests': min(5, batch_size),
                'timeout': 120
            }
            return cls.create_summarizer(SummarizerType.OPENAI_FAST, config_overrides)
        elif batch_size > 5:
            # Medium batch - use standard config
            config_overrides = {
                'concurrent_requests': min(3, batch_size)
            }
            return cls.create_summarizer(SummarizerType.OPENAI_STANDARD, config_overrides)
        else:
            # Small batch - use high quality config
            return cls.create_summarizer(SummarizerType.OPENAI_HIGH_QUALITY)
    
    @classmethod
    def create_for_quality(cls, provider: Optional[LLMProvider] = None) -> ISummarizer:
        """Create summarizer optimized for quality."""
        
        if provider == LLMProvider.ANTHROPIC:
            return cls.create_summarizer(SummarizerType.ANTHROPIC_STANDARD)
        elif provider == LLMProvider.AZURE_OPENAI:
            return cls.create_summarizer(SummarizerType.AZURE_OPENAI)
        else:
            return cls.create_summarizer(SummarizerType.OPENAI_HIGH_QUALITY)
    
    @classmethod
    def create_for_speed(cls, provider: Optional[LLMProvider] = None) -> ISummarizer:
        """Create summarizer optimized for speed."""
        
        if provider == LLMProvider.OLLAMA:
            return cls.create_summarizer(SummarizerType.OLLAMA_LOCAL)
        elif provider == LLMProvider.HUGGINGFACE:
            return cls.create_summarizer(SummarizerType.HUGGINGFACE_LOCAL)
        else:
            return cls.create_summarizer(SummarizerType.OPENAI_FAST)
    
    @staticmethod
    def _create_summarizer_from_config(config: SummarizationConfig) -> ISummarizer:
        """Create summarizer instance from configuration."""
        from .summarizer import Summarizer
        return Summarizer(config)
    
    @staticmethod
    def _get_api_key_from_env(provider: LLMProvider) -> Optional[str]:
        """Get API key from environment variables."""
        env_vars = {
            LLMProvider.OPENAI: ["OPENAI_API_KEY", "SUMMARIZER_API_KEY"],
            LLMProvider.ANTHROPIC: ["ANTHROPIC_API_KEY", "SUMMARIZER_API_KEY"],
            LLMProvider.AZURE_OPENAI: ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "SUMMARIZER_API_KEY"],
            LLMProvider.HUGGINGFACE: ["HUGGINGFACE_API_KEY", "SUMMARIZER_API_KEY"],
            LLMProvider.OLLAMA: []  # Ollama doesn't need API key
        }
        
        for env_var in env_vars.get(provider, []):
            key = os.getenv(env_var)
            if key:
                return key
        
        return None
    
    @staticmethod
    def _get_default_model(provider: LLMProvider) -> str:
        """Get default model for provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4-turbo-preview",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            LLMProvider.AZURE_OPENAI: "gpt-4",
            LLMProvider.HUGGINGFACE: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            LLMProvider.OLLAMA: "llama2"
        }
        return defaults.get(provider, "gpt-3.5-turbo")
    

    @staticmethod
    def _estimate_processing_speed(config: SummarizationConfig) -> str:
        """Estimate processing speed."""
        if config.llm_provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]:
            return "Variable (depends on hardware)"
        elif config.model_name == "gpt-3.5-turbo":
            return "Fast"
        elif "gpt-4" in config.model_name:
            return "Medium"
        else:
            return "Medium"
    
    @staticmethod
    def _estimate_quality(config: SummarizationConfig) -> str:
        """Estimate output quality."""
        if "gpt-4" in config.model_name or "claude-3" in config.model_name:
            return "High"
        elif config.model_name == "gpt-3.5-turbo":
            return "Medium"
        elif config.llm_provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]:
            return "Variable (depends on model)"
        else:
            return "Medium"
