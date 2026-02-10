# Audio Processor

A production-grade audio processing library with speaker diarization and transcription capabilities, designed for integration with the Sum-It-Up Agent ecosystem.

## Features

- **Speaker Diarization**: Identify different speakers in audio using PyAnnote
- **Transcription**: High-quality speech-to-text using Whisper models
- **Multiple Output Formats**: JSON, TXT, SRT, CSV export options
- **Factory Pattern**: Easy instantiation with preset configurations
- **Use Case Pattern**: Clean separation of business logic
- **Production Ready**: Error handling, logging, resource management
- **MCP Integration**: Model Context Protocol server for distributed processing

## Architecture

The library follows clean architecture principles with:

- **Interface**: Abstract base class defining the contract
- **Implementation**: Concrete audio processor with PyAnnote + Whisper
- **Factory**: Creates processors with different configurations
- **Use Case**: Business logic for file processing and exports

## Quick Start

```python
import os
from sum_it_up_agent.audio_processor import AudioProcessingUseCase, ProcessorType

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

### MCP Server Usage

```bash
# Start the MCP server
python -m sum_it_up_agent.audio_processor.mcp_server_audio

# Use with fastmcp client
from fastmcp import Client

async with Client("stdio://audio-processor") as client:
    result = await client.call_tool("process_audio_file", {
        "audio_path": "meeting.mp3",
        "preset": "high_quality",
        "output_format": "json"
    })
```

## Configuration Presets

- **STANDARD**: Balanced quality and speed (base model)
- **FAST**: Quick processing (tiny model, int8)
- **HIGH_QUALITY**: Best accuracy (large-v3 model)

## Custom Configuration

```python
from sum_it_up_agent.audio_processor import AudioProcessingConfig, DeviceType

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

- Python 3.11+
- CUDA-capable GPU (recommended)
- HuggingFace token for diarization

## Installation

```bash
poetry install
```

## Integration with Sum-It-Up Agent

This audio processor is designed to work seamlessly with the Sum-It-Up Agent:

- **MCP Server**: Run as independent service for distributed processing
- **Agent Integration**: Automatically called by the main agent for audio processing
- **Error Handling**: Graceful degradation when diarization fails
- **Format Support**: Outputs in agent-compatible JSON format

## Examples

See `examples/audio_processing_examples.py` for complete usage examples.
