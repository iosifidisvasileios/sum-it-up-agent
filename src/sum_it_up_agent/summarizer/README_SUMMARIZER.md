# Meeting Summarizer

A production-grade meeting summarization library using LLMs and structured prompt templates.

## Features

- **Multiple LLM Providers**: OpenAI, Anthropic, Azure OpenAI, HuggingFace, Ollama
- **Template-Based Summarization**: Uses structured prompts for different meeting types
- **Production Ready**: Error handling, logging, resource management, batch processing
- **Cost Estimation**: Estimate token usage and costs before processing
- **Multiple Output Formats**: JSON, CSV, TXT export options
- **Flexible Configuration**: Preset and custom configuration options

## Architecture

The library follows clean architecture principles with:

- **Interface**: Abstract base class defining the contract
- **Implementation**: Concrete summarizer with LLM integration
- **Factory**: Creates summarizers with different configurations
- **Use Case**: Business logic for file processing and analysis

## Quick Start

```python
from sum_it_up_agent.summarizer import SummarizationUseCase, SummarizerType

# Create use case with OpenAI
use_case = SummarizationUseCase.create_with_preset(
    summarizer_type=SummarizerType.OPENAI_STANDARD,
    api_key="your-openai-api-key"
)

# Summarize a transcription file
with use_case.summarizer:
    result = use_case.summarize_transcription_file(
        file_path="meeting_transcript.json",
        meeting_type="planning / coordination meeting",
        output_dir="./summaries"
    )

    if result.is_successful():
        print(f"Summary: {result.summary_data['executive_summary']}")
```

## Configuration Presets

- **OPENAI_FAST**: Fast processing with GPT-3.5
- **OPENAI_STANDARD**: Balanced performance with GPT-4
- **OPENAI_HIGH_QUALITY**: Best quality with GPT-4
- **ANTHROPIC_STANDARD**: Claude-3 for high-quality summaries
- **AZURE_OPENAI**: Azure-hosted GPT-4
- **HUGGINGFACE_LOCAL**: Local models via HuggingFace
- **OLLAMA_LOCAL**: Local models via Ollama
- **COST_OPTIMIZED**: Minimize costs for batch processing

## Supported Meeting Types

The library supports all meeting types from the templates package:

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

## Input Format

The library expects JSON files with transcription segments:

```json
{
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.5,
      "speaker": "SPEAKER_00",
      "text": "Hello everyone, let's start the meeting."
    }
  ]
}
```

Or audio processor output format:

```json
{
  "metadata": {
    "segments": [
      {
        "start_time": 0.0,
        "end_time": 3.5,
        "speaker": "SPEAKER_00",
        "text": "Hello everyone, let's start the meeting."
      }
    ]
  }
}
```

## Usage Examples

### Single File Summarization
```python
result = use_case.summarize_transcription_file(
    file_path="meeting.json",
    meeting_type="team status sync / standup"
)
```

### Batch Processing
```python
file_configs = [
    {"file_path": "meeting1.json", "meeting_type": "planning / coordination meeting"},
    {"file_path": "meeting2.json", "meeting_type": "decision-making meeting"}
]

results = use_case.summarize_multiple_files(file_configs, "./summaries")
```

### Audio Processor Integration
```python
result = use_case.summarize_from_audio_processor_output(
    audio_processor_output="transcript.json",
    meeting_type="planning / coordination meeting"
)
```


### Results Analysis
```python
analysis = use_case.analyze_summarization_results(results)
print(f"Success rate: {analysis['success_rate']:.2%}")
print(f"Average processing time: {analysis['avg_processing_time']:.2f}s")
```

## Environment Variables

Configure using environment variables:

```bash
export SUMMARIZER_PROVIDER=openai
export SUMMARIZER_MODEL=gpt-4-turbo-preview
export SUMMARIZER_API_KEY=your_api_key
export SUMMARIZER_TEMPERATURE=0.1
export SUMMARIZER_CONCURRENT_REQUESTS=2
```

Then create from environment:

```python
use_case = SummarizationUseCase.create_from_environment()
```

## LLM Provider Setup

### OpenAI
```python
config = SummarizationConfig(
    llm_provider=LLMProvider.OPENAI,
    model_name="gpt-4-turbo-preview",
    api_key="your-openai-api-key"
)
```

### Anthropic
```python
config = SummarizationConfig(
    llm_provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",
    api_key="your-anthropic-api-key"
)
```

### Azure OpenAI
```python
config = SummarizationConfig(
    llm_provider=LLMProvider.AZURE_OPENAI,
    model_name="gpt-4",
    api_key="your-azure-api-key",
    api_base="your-azure-endpoint",
    azure_deployment="your-deployment-name"
)
```

### Local Models (Ollama)
```python
config = SummarizationConfig(
    llm_provider=LLMProvider.OLLAMA,
    model_name="llama2",
    ollama_host="http://localhost:11434"
)
```

## Output Format

The summarizer returns structured JSON based on the meeting type template:

```json
{
  "meeting_type": "planning / coordination meeting",
  "title": "Project Planning Discussion",
  "time_range": {"start_sec": 0.0, "end_sec": 1800.0},
  "participants": ["SPEAKER_00", "SPEAKER_01"],
  "executive_summary": "The team discussed project timeline and deliverables...",
  "plan_overview": ["Phase 1: Requirements", "Phase 2: Development"],
  "decisions": [...],
  "action_items": [...],
  "risks": [...]
}
```

## Requirements

- Python 3.10+
- API keys for chosen LLM provider(s)
- Optional: OpenAI library (`pip install openai`)
- Optional: Anthropic library (`pip install anthropic`)
- Templates package for prompt templates

## Installation

```bash
poetry install
```

## Examples

See `examples/summarizer_examples.py` for complete usage examples including:

- Basic usage with different providers
- Custom configuration
- Batch processing
- Cost estimation
- Audio processor integration
- Environment configuration

## Performance Tips

- Use `SummarizerType.COST_OPTIMIZED` for large batches
- Use `SummarizerType.OPENAI_FAST` for quick summaries
- Adjust `concurrent_requests` based on API limits
- Estimate costs before processing large batches
- Use local models (Ollama/HuggingFace) for privacy-sensitive data
