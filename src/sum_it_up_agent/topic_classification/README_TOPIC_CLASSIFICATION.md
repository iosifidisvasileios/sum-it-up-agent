# Topic Classification

A production-grade zero-shot topic classification library for conversations using transformer models.

## Features

- **Zero-Shot Classification**: Classify conversations without training data
- **Multiple Models**: Ensemble of state-of-the-art transformer models
- **Flexible Configuration**: Preset and custom configuration options
- **Batch Processing**: Efficient processing of multiple files
- **Multiple Output Formats**: JSON, CSV, TXT export options
- **Performance Optimization**: Device-specific optimizations
- **Production Ready**: Error handling, logging, resource management

## Architecture

The library follows clean architecture principles with:

- **Interface**: Abstract base class defining the contract
- **Implementation**: Concrete classifier with transformer models
- **Factory**: Creates classifiers with different configurations
- **Use Case**: Business logic for file processing and analysis

## Quick Start

```python
from sum_it_up_agent.topic_classification import TopicClassificationUseCase, ClassifierType

# Create use case with standard preset
use_case = TopicClassificationUseCase.create_with_preset(
    classifier_type=ClassifierType.STANDARD
)

# Classify a conversation file
with use_case.classifier:
    result = use_case.classify_single_file("conversation.json")

    print(f"Topic: {result.predicted_topic}")
    print(f"Confidence: {result.confidence:.3f}")
```

## Configuration Presets

- **FAST**: Single lightweight model for quick processing
- **STANDARD**: Balanced performance with 2 models
- **HIGH_ACCURACY**: Best accuracy with 3 models
- **LIGHTWEIGHT**: Optimized for CPU processing

## Custom Configuration

```python
from sum_it_up_agent.topic_classification import TopicClassificationConfig, DeviceType

config = TopicClassificationConfig(
    device=DeviceType.CUDA,
    models=["FacebookAI/roberta-large-mnli"],
    first_n_segments=10,
    hypothesis_template="This conversation is about {}."
)

use_case = TopicClassificationUseCase.create_with_custom_classifier(config)
```

## Default Topics

The classifier comes with default conversation topics:

- team status sync / standup
- planning / coordination meeting
- decision-making meeting
- brainstorming session
- retrospective / postmortem
- training / onboarding
- interview
- customer call / sales demo
- support / incident call
- other

## Usage Examples

### Single File Classification
```python
result = use_case.classify_single_file("meeting.json")
```

### Batch Processing
```python
results = use_case.classify_multiple_files(["file1.json", "file2.json"])
```

### Directory Processing
```python
results = use_case.classify_directory("./transcriptions", "*.json")
```

### Results Analysis
```python
analysis = use_case.analyze_classification_results(results)
print(f"Most common topic: {analysis['most_common_topic']}")
```

### Export Results
```python
use_case.export_results(results, "results.json", "json", include_analysis=True)
```

## Input Format

The library expects JSON files with a "segments" array containing conversation segments:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "Hello everyone, let's start the meeting.",
      "start_time": 0.0,
      "end_time": 3.5
    }
  ]
}
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for performance)
- Transformers library
- PyTorch

## Installation

```bash
poetry install
```

## Examples

See `examples/topic_classification_examples.py` for complete usage examples including:

- Basic usage with presets
- Custom configuration
- Batch processing
- Directory processing
- Performance optimization
- Notebook equivalent examples

## Performance Tips

- Use `ClassifierType.FAST` for large batches
- Use `ClassifierType.HIGH_ACCURACY` for single important files
- Use `DeviceType.CPU` if no GPU is available
- Adjust `first_n_segments` to balance accuracy vs speed
