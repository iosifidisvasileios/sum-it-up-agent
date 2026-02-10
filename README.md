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
Agentic AI Agent
├── Audio Processor MCP Server
├── Topic Classification MCP Server  
├── Summarizer MCP Server
└── Communicator MCP Server
```

### Why MCP Changes Everything

- **True Modularity**: Each component is an independent server that can be developed, tested, and deployed separately
- **Unlimited Extensibility**: Add new capabilities by creating new MCP servers without touching existing code
- **Universal Interoperability**: Standardized protocol works across languages, platforms, and cloud providers
- **Independent Scaling**: Scale individual components based on their specific resource needs
- **Isolated Testing**: Test each component in complete isolation with mocking and stubbing

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

### Core Technologies
- **Python 3.11+**: Modern Python with type hints and async support
- **Poetry 2.2.1**: Dependency management and packaging
- **PyTorch 2.0+**: GPU-accelerated deep learning
- **Transformers**: State-of-the-art transformer models
- **PyAnnote.audio**: Advanced speaker diarization
- **Faster-Whisper**: Optimized speech transcription

### AI/ML Infrastructure
- **HuggingFace Hub**: Model repository and management
- **Unsloth**: Optimized LLM inference
- **BitsAndBytes**: Efficient quantization
- **Scikit-learn**: Machine learning utilities
- **FastMCP**: Model Context Protocol implementation

### LLM Integration
- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo
- **Anthropic**: Claude-3 family
- **Azure OpenAI**: Enterprise-grade OpenAI
- **Local Models**: Ollama, HuggingFace models
- **Cost Optimization**: Smart token management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sum-it-up-agent.git
cd sum-it-up-agent

# Install dependencies
poetry install

# Set up environment variables (copy .env.example to .env and configure)
cp .env.example .env
# Edit .env with your API keys and MCP server settings
```

### Basic Agent Usage

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

### Component Usage

```python
from sum_it_up_agent.audio_processor import AudioProcessingUseCase, ProcessorType
from sum_it_up_agent.topic_classification import TopicClassificationUseCase, ClassifierType
from sum_it_up_agent.summarizer import SummarizationUseCase, SummarizerType

# Process audio file
audio_use_case = AudioProcessingUseCase.create_with_preset(ProcessorType.HIGH_QUALITY)
with audio_use_case.processor:
  segments = audio_use_case.process_audio_file("meeting.mp3", output_format="json")

# Classify meeting type
topic_use_case = TopicClassificationUseCase.create_with_preset(ClassifierType.STANDARD)
with topic_use_case.classifier:
  result = topic_use_case.classify_single_file("meeting_transcription.json")

# Generate intelligent summary
summary_use_case = SummarizationUseCase.create_with_preset(SummarizerType.OPENAI_STANDARD)
with summary_use_case.summarizer:
  summary = summary_use_case.summarize_transcription_file(
    "meeting_transcription.json",
    meeting_type=result.predicted_topic
  )
```

### MCP Server Usage

```bash
# Start MCP servers (in separate terminals)
python src/audio_processor/mcp_server_audio.py
python src/topic_classification/mcp_topic_classification.py  
python src/summarizer/mcp_summarizer.py
python src/communicator/mcp_communicator.py

# Use with Agentic AI Agent
from fastmcp import Client

async with Client("http://localhost:9000/mcp_audio_processor") as client:
    result = await client.call_tool("process_audio_file", {
        "audio_path": "meeting.mp3",
        "preset": "high_quality",
        "output_format": "json"
    })
```

## Use Cases

### Enterprise Meetings
- **Planning Sessions**: Action items, decisions, timelines
- **Retrospectives**: Lessons learned, improvement opportunities  
- **Decision Making**: Options analysis, final decisions, rationale
- **Customer Calls**: Requirements, feedback, next steps

### Education & Training
- **Lectures**: Key concepts, summaries, study materials
- **Training Sessions**: Learning objectives, skill assessments
- **Interviews**: Candidate evaluation, key insights

### Research & Development
- **Brainstorming**: Idea generation, concept development
- **Technical Discussions**: Architecture decisions, trade-offs
- **Lab Meetings**: Experimental results, next steps

### Advanced Agent Capabilities
- **Natural Language Understanding**: Parse complex user requests with context
- **Multi-Channel Communication**: Email, Slack, Discord, Telegram, Jira integration
- **Custom Summary Types**: Action items, decisions, key points, executive summaries
- **Flexible Configuration**: Customize every aspect of the processing pipeline

## Key Differentiators

### Intelligent Agent Architecture
Unlike simple pipeline orchestrators, Sum-It-Up Agent:
- **Understands** natural language requests with LLM-powered prompt parsing
- **Reasons** about user intent and extracts structured requirements
- **Adapts** processing strategy based on content and communication needs
- **Integrates** with multiple communication channels and LLM providers
- **Configures** itself dynamically based on user preferences

### MCP Architecture
- **Modular**: Each component is an independent, replaceable service
- **Extensible**: Add new capabilities without modifying existing code
- **Interoperable**: Works with any MCP-compliant service
- **Scalable**: Scale components independently based on needs

### Production-Ready
- **Error Handling**: Comprehensive error recovery and reporting
- **Monitoring**: Detailed logging and performance metrics
- **Configuration**: Flexible preset and custom configurations
- **Testing**: Extensive test coverage and validation

## Development

### Architecture Overview

```
src/
├── sum_it_up_agent/         # Main package
│   ├── agent/              # Main AI agent with LLM-based parsing
│   ├── audio_processor/    # Audio processing with diarization
│   ├── topic_classification/ # Zero-shot topic classification  
│   ├── summarizer/         # LLM-powered summarization
│   ├── templates/         # Structured prompt templates
│   └── communicator/      # Multi-channel communication
└── examples/              # Usage examples and tutorials
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

## Roadmap

### Coming Soon
- **Multi-Language Support**: Transcription and summarization in 50+ languages
- **Real-Time Processing**: Live meeting transcription and analysis
- **Advanced Agent Capabilities**: Multi-step reasoning and tool usage
- **Cloud MCP Services**: Managed MCP servers for enterprise deployment

### Future Vision
- **AGI Integration**: Advanced reasoning and planning capabilities
- **Mobile Apps**: Native iOS and Android applications
- **Plugin Ecosystem**: Third-party MCP server marketplace
- **Enterprise Features**: SSO, audit logs, compliance

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## CI/CD

This repo uses **GitHub Actions** for a lightweight but robust CI/CD pipeline.

### CI (Pull Requests + main)

Workflow: `.github/workflows/ci.yml`

- **Validates packaging** with `poetry check`
- **Builds artifacts** with `poetry build`
- **Runs a syntax sanity check** via `python -m compileall -q src`

Note: the CI is intentionally kept lightweight and does **not** install the full ML dependency stack.

### CD (Release + Publish)

Workflow: `.github/workflows/release.yml`

Trigger a release by pushing a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This workflow:

- Builds `sdist` + `wheel`
- Creates a **GitHub Release** and uploads the artifacts
- Publishes to **PyPI** using **trusted publishing** (OIDC)

#### PyPI configuration (one-time)

To enable trusted publishing, configure your PyPI project to trust this GitHub repo/workflow (no API token required). In GitHub, the publish job runs in an environment named `pypi`.

### Important: Dual-Licensing Model
This project uses a dual-license model to keep development sustainable while remaining free for non-commercial use. All contributors must accept our [Contributor License Agreement (CLA)](CLA.md) to maintain this model.

### Pull Request Process
- Use our [PR template](.github/pull_request_template.md) for all submissions
- Ensure all CLA requirements are met
- Follow our code review guidelines

### Areas for Contribution
- **New MCP Servers**: Add new capabilities via MCP
- **Language Support**: Add transcription for new languages
- **Templates**: Create new meeting type templates
- **Testing**: Improve test coverage and add integration tests
- **Documentation**: Improve docs and create tutorials

## License

This project is offered under a **dual-license model**:

### Non-Commercial License (FREE)
- Free for personal, educational, and research use
- Open source with full access to all features
- See [LICENSE-NONCOMMERCIAL](LICENSE-NONCOMMERCIAL) for complete terms

### Commercial License (PAID)
- Required for any commercial use, including:
  - For-profit business operations
  - SaaS/hosted services
  - Commercial products or services
  - Paid consulting or training
- Contact: **billiosifidis@gmail.com** | **v-iosifidis.com**
- See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details

### Contributor License Agreement
All contributions require acceptance of our [CLA](CLA.md) to maintain dual-licensing model.

Unsure about your use case? Contact us for clarification!

## Acknowledgments

- **HuggingFace** for amazing transformer models and tools
- **OpenAI** for powerful language models
- **Anthropic** for Claude models
- **Ollama** for local LLM inference and privacy-focused AI
- **FastMCP** for the Model Context Protocol implementation
- **PyAnnote** for speaker diarization technology

## Contact

- **Email**: billiosifidis@gmail.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/sum-it-up-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sum-it-up-agent/discussions)

---

## Ready to Transform Your Meeting Intelligence?

**Sum-It-Up Agent** represents the future of **Agentic AI** - systems that don't just process data, but **think, reason, and make intelligent decisions**. With our revolutionary **MCP-based architecture**, you're not just getting a tool - you're getting a platform that can evolve and adapt with your needs.

**Star this repo** to follow our journey toward truly intelligent meeting processing!

**Try it now** and experience the difference that true Agentic AI can make in your meeting workflow!
