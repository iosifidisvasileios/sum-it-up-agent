# Sum-It-Up Agent: Agentic AI Meeting Intelligence Platform

> Next-generation AI Agent architecture powered by Model Context Protocol (MCP) for intelligent meeting processing and analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-2.2.1+-60A5FA.svg)](https://python-poetry.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-3.0+-green.svg)](https://fastmcp.com)
[![License](https://img.shields.io/badge/License-Dual%20License-yellow.svg)](LICENSE)

## What Makes This Revolutionary?

**Sum-It-Up Agent** isn't just another transcription tool - it's a **true Agentic AI system** that thinks, reasons, and makes autonomous decisions about how to process your meetings. Built on the cutting-edge **Model Context Protocol (MCP)**, it represents the future of modular, extensible AI architectures.

### Agentic AI Capabilities

- **Autonomous Decision-Making**: The agent analyzes audio characteristics and chooses optimal processing strategies
- **Intelligent Reasoning**: Adapts its approach based on content complexity, user requirements, and constraints
- **Dynamic Planning**: Creates and executes plans that can adapt in real-time to changing conditions
- **Smart Optimization**: Balances speed, quality, and cost based on your priorities
- **User-Aware Processing**: Intelligently incorporates custom instructions and preferences

## MCP-Based Architecture

```
Sum-It-Up App (Singleton)
├── Audio Processing Agent (Orchestrator)
│   ├── Audio Processor MCP Server (Port 9001)
│   ├── Topic Classification MCP Server (Port 9002)
│   ├── Summarizer MCP Server (Port 9000)
│   └── Communicator MCP Server (Port 9003)
└── Interactive Interface & Server Management
```

### Architecture Benefits

- **Singleton Management**: The app manages all MCP servers as a single unit
- **Environment Isolation**: Each server gets only the environment variables it needs
- **Independent Lifecycle**: Servers persist even when agent instances are destroyed
- **Clean Separation**: App handles infrastructure, agent handles processing pipeline
- **Graceful Scaling**: Start/stop all servers together with proper health checks
- **True Modularity**: Each component is an independent server with clean boundaries
- **Unlimited Extensibility**: Add new MCP servers without touching existing code
- **Universal Interoperability**: Standardized protocol works across all platforms

## Core Capabilities

### Audio Processing
- **Multi-Format Support**: MP3, MP4, WAV, M4A, FLAC, and more
- **Speaker Diarization**: Automatic speaker identification and separation
- **High-Quality Transcription**: State-of-the-art Whisper models with GPU acceleration
- **Format Conversion**: Intelligent audio format optimization
- **Batch Processing**: Handle multiple files efficiently with parallel processing

### Topic Classification
- **Zero-Shot Learning**: Classify meetings without training data
- **Ensemble Methods**: Multiple models working together for improved accuracy
- **Meeting Type Detection**: Automatically identify planning sessions, retrospectives, interviews, and more
- **Confidence Scoring**: Reliable classification with uncertainty quantification

### Intelligent Summarization
- **Document-as-Implementation Prompts**: All LLM prompts are editable `.txt` files—no code changes required
- **Template-Based**: Structured summaries tailored to meeting types
- **Multiple LLM Providers**: OpenAI GPT-4, Anthropic Claude, local Ollama models
- **Custom Instructions**: User-aware summarization that adapts to your specific needs
- **Cost Optimization**: Smart token usage and cost estimation
- **Multi-Format Output**: JSON, CSV, TXT, and custom formats

### Advanced Agent Features
- **LLM-Based Prompt Parsing**: Intelligent understanding of natural language requests
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama for intent detection
- **Dynamic Planning**: Creates and executes plans that adapt to requirements
- **Comprehensive Configuration**: Flexible settings for all components
- **Error Handling**: Graceful degradation and detailed reporting

## Technology Stack

- **Python 3.11+**: Modern Python with type hints and async support
- **Poetry 2.2.1**: Dependency management and packaging
- **PyTorch 2.0+**: GPU-accelerated deep learning
- **Transformers**: State-of-the-art transformer models
- **FastMCP**: Model Context Protocol implementation
- **HuggingFace Hub**: Model repository and management
- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo
- **Anthropic**: Claude-3 family
- **Azure OpenAI**: Enterprise-grade OpenAI
- **Local Models**: Ollama, HuggingFace models

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sum-it-up-agent.git
cd sum-it-up-agent

# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and MCP server settings

# Run the interactive app
python -m src.sum_it_up_agent
```

### System Requirements

- **Python**: 3.11+ (recommended 3.11)
- **Operating System**: Linux, macOS, or Windows
- **GPU**: CUDA-compatible GPU (recommended for optimal performance)
- **RAM**: 16GB+ recommended for large audio files
- **Storage**: 10GB+ free space for models and temporary files
- **Network**: Stable internet connection for LLM API calls

### Key Dependencies

- **PyTorch** 2.9.0: GPU-accelerated deep learning
- **Transformers** 4.57.6: HuggingFace model library
- **FastMCP** 3.0.0b2+: Model Context Protocol implementation
- **PyAnnote.audio** 3.1.0+: Speaker diarization and transcription
- **Faster-Whisper** 1.2.1: Optimized speech recognition
- **OpenAI** 2.16.0+: GPT model access
- **BitsAndBytes** 0.49.0: Model quantization
- **Unsloth** 3.5.0: Optimized LLM inference

## Advanced Usage

For programmatic access, you can use the agent directly:

```python
import asyncio
from sum_it_up_agent.agent import AudioProcessingAgent, AgentConfig

async def process_meeting():
    config = AgentConfig()
    
    async with AudioProcessingAgent(config) as agent:
        result = await agent.process_request(
            "meeting.mp3",
            "Please summarize and send action points to john@example.com"
        )
        
        if result.success:
            print(f"Success! Summary saved to: {result.summary_file}")
        else:
            print(f"Failed: {result.error_message}")

asyncio.run(process_meeting())
```

## Sum-It-Up App Usage

### Interactive Mode (Recommended)

```bash
# Run the app
python -m src.sum_it_up_agent
```

### Command Line Mode

```bash
# Process a single file
python -m src.sum_it_up_agent.app /path/to/audio.mp4 "summarize this meeting and email it to user@example.com"
```

### App Features

- **Automatic Server Booting**: Starts all MCP servers (audio processor, topic classifier, summarizer, communicator)
- **Interactive Mode**: Type commands interactively
- **Health Checks**: Waits for servers to be ready before processing
- **Graceful Shutdown**: Properly stops all servers on exit
- **Environment Security**: Each server gets only the environment variables it needs

### Example Interactive Session

```
🚀 Starting Sum-It-Up Agent...
==================================================
📡 Booting MCP servers...
✅ All servers ready!
🔗 Initializing clients...
✅ Ready to process!
==================================================
🎯 Sum-It-Up Agent - Interactive Mode
Type 'quit' or 'exit' to stop
==================================================

📁 Enter audio file path: /home/user/meeting.mp4
💬 What would you like to do with this audio? summarize this meeting and email it to user@example.com

⚡ Processing /home/user/meeting.mp4...
📝 Request: summarize this meeting and email it to user@example.com
------------------------------
✅ Processing completed successfully!
📄 Transcription: /tmp/meeting_transcription.json
📋 Summary: /tmp/meeting_summary.json
📧 Communication sent:
  ✅ email
⏱️  Total time: 45.23s
==================================================
```

### Stopping the App

- Type `quit` or `exit` in interactive mode
- Press `Ctrl+C` to interrupt
- The app will automatically shutdown all servers gracefully

## Use Cases

- **Enterprise Meetings**: Planning sessions, retrospectives, decision making, customer calls
- **Education & Training**: Lectures, training sessions, interviews
- **Research & Development**: Brainstorming, technical discussions, lab meetings
- **Custom Summaries**: Action items, decisions, key points, executive summaries

## Key Differentiators

### Intelligent Agent Architecture
- **Natural Language Understanding**: Parse complex user requests with LLM-powered prompt parsing
- **Reasoning & Adaptation**: Adapts processing strategy based on content and communication needs
- **Multi-Channel Integration**: Email, Slack, Discord, Telegram, Jira integration
- **Dynamic Configuration**: Configures itself based on user preferences

### MCP Architecture
- **Modular**: Each component is an independent, replaceable service
- **Extensible**: Add new capabilities without modifying existing code
- **Interoperable**: Works with any MCP-compliant service
- **Scalable**: Scale components independently based on needs
- **Production-Ready**: Comprehensive error handling, monitoring, and testing

## Development

### Architecture Overview

```
src/
├── sum_it_up_agent/         # Main package
│   ├── agent/              # Main AI agent with LLM-based parsing
│   ├── audio_processor/    # Audio processing with diarization
│   ├── topic_classification/ # Zero-shot topic classification  
│   ├── summarizer/         # LLM-powered summarization
│   ├── templates/         # Structured prompt templates (file-backed)
│   │   └── prompt_files/  # All LLM prompts as editable .txt files
│   └── communicator/      # Multi-channel communication
└── examples/              # Usage examples and tutorials
```

### Prompt Customization (Document-as-Implementation)

All LLM prompts are stored as editable `.txt` files under `src/sum_it_up_agent/templates/prompt_files/`:

- **Meeting prompts**: `prompt_files/meeting/*.txt`
  - `team_status_sync_standup.txt`
  - `planning_coordination_meeting.txt`
  - `decision_making_meeting.txt`
  - `brainstorming_session.txt`
  - `retrospective_postmortem.txt`
  - `training_onboarding.txt`
  - `interview.txt`
  - `customer_call_sales_demo.txt`
  - `support_incident_call.txt`
  - `other.txt`

- **System prompts**: `prompt_files/system/*.txt`
  - `intent_extraction.txt` – Used by the agent to parse user requests
  - `structured_json_assistant.txt` – Used by OpenAI/Azure summarizers

You can edit any `.txt` file to change behavior without touching code. The system loads prompts at runtime via `importlib.resources`.

## PromptParser Evaluation Harness

The project includes a comprehensive evaluation harness for the `PromptParser` component to benchmark accuracy and latency across different LLM models and system prompts.

### Features

- **Dataset-Driven Testing**: Curated prompts with expected `UserIntent` fields (channels, summary types, recipients, custom instructions)
- **Multiple Models**: Test against any Ollama model (e.g., Mistral, Llama, Gemma)
- **System Prompt Variants**: Compare default vs strict JSON system prompts
- **Fair Latency Mode**: Optional cold-start benchmarking with model unload and cooldown
- **Aggregated Reporting**: Markdown tables with pass/fail rates and latency percentiles (avg/p50/p95)
- **Enum Validation**: Ensures parsed channels and types match `CommunicationChannel` and `SummaryType` enums
- **Recipient Sanity Checks**: Validates email format where expected

### Quick Start

Run the full evaluation matrix:

```bash
python -m unittest -v tests.test_prompt_parser_eval
```

#### Environment Controls

```bash
# Models to test (comma-separated)
PROMPT_EVAL_MODELS="hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL,hf.co/unsloth/gemma-3-27b-it-GGUF:Q2_K_XL"

# System prompt variants to test
PROMPT_EVAL_SYSTEM_PROMPTS="default,strict_json"

# Enable fair-latency (cold-start) mode
PROMPT_EVAL_FAIR_LATENCY=1

# Cooldown between cases (ms) to let Ollama release VRAM
PROMPT_EVAL_COOLDOWN_MS=500

# Write report to file
PROMPT_EVAL_REPORT_PATH=prompt_parser_eval_report.md
```

### Example Script

For quick manual testing:

```bash
python examples/prompt_parser_example.py
```

### Report Output

The evaluation prints a Markdown report with four tables:
- Summary by model and system prompt
- Summary by model (aggregated across prompts)
- Summary by system prompt (aggregated across models)
- Summary by test case (hardest prompts)

Example excerpt:

```markdown
| model | system_prompt | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL | default | 12 | 12 | 0 | 100.0% | 621.3 | 598.1 | 789.4 |
| hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL | strict_json | 12 | 11 | 1 | 91.7% | 587.2 | 562.0 | 712.5 |
```

### Agent Components
- **Main Agent**: `src/sum_it_up_agent/agent/orchestrator.py`
- **Prompt Parser**: LLM-based intent detection and parsing
- **Pipeline Manager**: Orchestrates all processing steps
- **Error Handler**: Comprehensive error management and recovery

### MCP Servers
- **Audio Processor**: `src/audio_processor/mcp_server_audio.py`
- **Topic Classification**: `src/topic_classification/mcp_topic_classification.py`
- **Summarizer**: `src/summarizer/mcp_summarizer.py`
- **Communicator**: `src/communicator/mcp_communicator.py`

### Examples & Tutorials
- **Agent Usage**: `examples/agent_example.py`
- **Audio Processing**: `examples/audio_processing_examples.py`
- **MCP Integration**: `examples/mcp_audio_testing.ipynb`
- **Summarization**: `examples/summarizer_examples.py`
- **Classification**: `examples/topic_classification_examples.py`
- **PromptParser**: `examples/prompt_parser_example.py`
- **PromptParser Evaluation**: `tests/test_prompt_parser_eval.py`

## Roadmap

### Coming Soon
- **Multi-Language Support**: Transcription and summarization in 50+ languages
- **Advanced Agent Capabilities**: Multi-step reasoning and tool usage
- **Cloud MCP Services**: Managed MCP servers for enterprise deployment

### Future Vision
- **AGI Integration**: Advanced reasoning and planning capabilities
- **Mobile Apps**: Native iOS and Android applications
- **Plugin Ecosystem**: Third-party MCP server marketplace
- **Enterprise Features**: SSO, audit logs, compliance

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution
- **New MCP Servers**: Add new capabilities via MCP
- **Language Support**: Add transcription for new languages
- **Templates**: Create new meeting type templates by adding `.txt` files under `src/sum_it_up_agent/templates/prompt_files/meeting/`
- **Testing**: Improve test coverage and add integration tests
- **Documentation**: Improve docs and create tutorials

## License

This project is offered under a **dual-license model**:

### Non-Commercial License (FREE)
- Free for personal, educational, and research use
- Open source with full access to all features
- See [LICENSE-NONCOMMERCIAL](LICENSE-NONCOMMERCIAL) for complete terms

### Commercial License (PAID)
- Required for any commercial use, including for-profit business operations, SaaS/hosted services, commercial products or services, and paid consulting or training
- Contact: **billiosifidis@gmail.com** | **v-iosifidis.com**
- See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details

All contributions require acceptance of our [CLA](CLA.md) to maintain dual-licensing model.

## Contact

- **Email**: billiosifidis@gmail.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/sum-it-up-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sum-it-up-agent/discussions)
