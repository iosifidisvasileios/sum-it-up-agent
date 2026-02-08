# Audio Processor

A production-grade audio processing library with speaker diarization and transcription capabilities.

## Features

- **Speaker Diarization**: Identify different speakers in audio using PyAnnote
- **Transcription**: High-quality speech-to-text using Whisper models
- **Multiple Output Formats**: JSON, TXT, SRT, CSV export options
- **Factory Pattern**: Easy instantiation with preset configurations
- **Use Case Pattern**: Clean separation of business logic
- **Production Ready**: Error handling, logging, resource management

## Architecture

The library follows clean architecture principles with:

- **Interface**: Abstract base class defining the contract
- **Implementation**: Concrete audio processor with PyAnnote + Whisper
- **Factory**: Creates processors with different configurations
- **Use Case**: Business logic for file processing and exports

## Quick Start

```python
import os
from src.audio_processor import AudioProcessingUseCase, ProcessorType

# Set your HuggingFace token
os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"

# Create use case with standard preset
use_case = AudioProcessingUseCase.create_with_preset(
    processor_type=ProcessorType.STANDARD,
    huggingface_token=os.getenv("HUGGINGFACE_TOKEN")
)

# Process audio file
with use_case.processor:
    segments = use_case.process_audio_file(
        audio_path="path/to/audio.mp3",
        output_format="json",
        save_to_file=True
    )
    
    # Get summary
    summary = use_case.get_transcription_summary(segments)
    print(f"Processed {summary['total_segments']} segments")
```

## Configuration Presets

- **STANDARD**: Balanced quality and speed (base model)
- **FAST**: Quick processing (tiny model, int8)
- **HIGH_QUALITY**: Best accuracy (large-v3 model)

## Custom Configuration

```python
from src.audio_processor import AudioProcessingConfig, DeviceType

config = AudioProcessingConfig(
    device=DeviceType.CUDA,
    whisper_model="large-v3",
    compute_type="float16",
    huggingface_token="your_token",
    merge_gap=0.5
)

use_case = AudioProcessingUseCase.create_with_custom_processor(config)
```

## Output Formats

- **JSON**: Complete data with metadata
- **TXT**: Simple readable format
- **SRT**: Subtitle format for videos
- **CSV**: Spreadsheet-compatible format

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- HuggingFace token for diarization

## Installation

```bash
poetry install
```

## Examples

See `examples/audio_processing_examples.py` for complete usage examples.
