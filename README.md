# Sum-It-Up Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org) [![Poetry](https://img.shields.io/badge/Poetry-2.2.1+-60A5FA.svg)](https://python-poetry.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org) [![FastMCP](https://img.shields.io/badge/FastMCP-3.0+-green.svg)](https://fastmcp.com) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
Agentic meeting intelligence built on the **Model Context Protocol (MCP)**: ingest audio/video → (optional) diarize → transcribe → classify → summarize → deliver.

---

## Table of Contents

- [Introduction](#introduction)
- [Why Sum-It-Up Agent?](#why-sum-it-up-agent)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [MCP Architecture](#mcp-architecture)
- [Core Capabilities](#core-capabilities)
- [Prompt Customization](#prompt-customization)
- [Programmatic Usage](#programmatic-usage)
- [PromptParser Evaluation Harness](#promptparser-evaluation-harness)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

**Sum-It-Up Agent** is a modular, MCP-native system for turning meeting recordings into structured outputs (summaries, key points, decisions, action items) and delivering them through communication channels.

The project is designed around **independent services** with clear boundaries, so you can evolve transcription, classification, summarization, and delivery without coupling everything into one monolith.

---

## Why Sum-It-Up Agent?

1. **MCP-first modularity**: each major capability is an MCP server you can swap or extend independently.
2. **Agentic orchestration**: LLM-based intent parsing + dynamic planning to decide what to run and how to format output.
3. **Document-as-implementation prompts**: prompts live as editable `.txt` files and load at runtime—no code edits needed.
4. **Multi-provider LLM support**: use hosted models or local inference depending on cost/latency/privacy constraints.
5. **Operational separation**: environment isolation per server and an app-level lifecycle manager (start/health-check/stop).
6. **Built-in evaluation for intent parsing**: benchmark accuracy + latency across models and prompt variants.

---

## Quick Start

### 1) Install

```bash
git clone https://github.com/yourusername/sum-it-up-agent.git
cd sum-it-up-agent
poetry install
```

### 2) Configure environment

```bash
cp .env.example .env
# edit .env with API keys and MCP server settings
```

**Key environment variables:**
- `SLACK_WEBHOOK_URL`: Slack incoming webhook URL for message delivery
- `SENDER_EMAIL_ACCOUNT`: Email account for sending summaries
- `SENDER_EMAIL_PASSWORD`: Email password or app password
- `SUM_IT_UP_LLM_MODEL`: LLM model for intent parsing
- MCP server URLs and ports (pre-configured)

### 3) Run (interactive mode)

```bash
python -m src.sum_it_up_agent
```

Or run a single command:

```bash
python -m src.sum_it_up_agent.app /path/to/audio.mp4 "summarize this meeting and email it to user@example.com and send to slack"
```

---

## How It Works

At runtime, the agent performs a pipeline like:

1. **Ingest & normalize** audio/video
2. **(Optional) diarization** for speaker separation
3. **Transcription** (Whisper-family models)
4. **Meeting-type classification** (zero-shot / ensembles)
5. **Instruction-aware summarization** using meeting templates + user instructions
5. **Delivery** via communicator (email, Slack, PDF export; other channels can be added)

Outputs can be saved as structured artifacts (e.g., JSON) for downstream workflows.

---

## MCP Architecture

```
Sum-It-Up App (Singleton)
├── Audio Processing Agent (Orchestrator)
│   ├── Audio Processor MCP Server (Port 9001)
│   ├── Topic Classification MCP Server (Port 9002)
│   ├── Summarizer MCP Server (Port 9000)
│   └── Communicator MCP Server (Port 9003)
└── Interactive Interface & Server Management
```

### Architecture highlights

- **Single lifecycle manager**: the app starts/stops all servers and validates readiness with health checks.
- **Environment scoping**: each server receives only the environment it needs.
- **Replaceable components**: each server is a clean integration boundary (e.g., swap diarization, change summarizer backend).
- **Scale by service**: scale or optimize transcription independently from summarization or delivery.

---

## Core Capabilities

### Audio processing
- Multi-format input (MP3, MP4, WAV, M4A, FLAC, …)
- Optional speaker diarization
- High-quality transcription with GPU acceleration where available
- Batch-friendly workflows (parallelizable stages)

### Topic classification
- Zero-shot meeting type detection (no training data required)
- Confidence scoring / uncertainty-aware output
- Support for common meeting classes (planning, retro, interview, support call, etc.)

### Summarization
- File-backed prompt templates for meeting-specific structure
- Multiple summary types (standard, action items, decisions, key points, executive, …)
- Multi-provider backends (hosted and local)
- Output formats: JSON, TXT, CSV (and extensible)

### Communication
- **Email delivery** with HTML-formatted summaries
- **Slack integration** via webhook URLs with professional formatting
- **PDF export** for archival and sharing
- **Extensible design** for adding new channels (Discord, Teams, etc.)

### Agent features
- LLM-based prompt parsing (turn “what I want” into a structured intent)
- Dynamic planning based on requested outputs and constraints
- Detailed error reporting and graceful degradation

---

## Prompt Customization

All prompts are plain `.txt` files under:

```
src/sum_it_up_agent/templates/prompt_files/
├── meeting/
└── system/
```

### Meeting templates (examples)
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

### System prompts
- `intent_extraction.txt` (intent parsing)
- `structured_json_assistant.txt` (structured output guidance)

Prompts are loaded at runtime (via packaged resources), so behavior can be changed without modifying Python code.

---

## Programmatic Usage

```python
import asyncio
from sum_it_up_agent.agent import AudioProcessingAgent, AgentConfig

async def process_meeting():
    config = AgentConfig()

    async with AudioProcessingAgent(config) as agent:
        result = await agent.process_request(
            "meeting.mp3",
            "Please summarize and send action points to john@example.com and also post to Slack"
        )

        if result.success:
            print(f"Summary saved to: {result.summary_file}")
            print(f"Communication results: {result.communication_results}")
        else:
            print(f"Failed: {result.error_message}")

asyncio.run(process_meeting())
```

---

## PromptParser Evaluation Harness

The repository includes a benchmarking harness to evaluate `PromptParser` quality and latency across models and prompt variants.

### What it measures
- Dataset-driven correctness for parsed intent fields (channels, summary types, recipients, custom instructions)
- Enum validation against `CommunicationChannel` and `SummaryType`
- Latency aggregates (avg / p50 / p95)
- Optional cold-start “fair latency” mode for more realistic local-model benchmarking

### Run
```bash
python -m unittest -v tests.test_prompt_parser_eval
```

### Environment controls
```bash
PROMPT_EVAL_MODELS="modelA,modelB"
PROMPT_EVAL_SYSTEM_PROMPTS="default,strict_json"
PROMPT_EVAL_FAIR_LATENCY=1
PROMPT_EVAL_COOLDOWN_MS=500
PROMPT_EVAL_REPORT_PATH=prompt_parser_eval_report.md
```

---

## Development

### Repo Layout

```
sum-it-up-agent/
├── src/
│   └── sum_it_up_agent/
│       ├── agent/
│       ├── audio_processor/
│       ├── topic_classification/
│       ├── summarizer/
│       ├── templates/
│       └── communicator/
└── tests/                      # Test suite and examples
    ├── examples/               # Usage examples and sample outputs
    └── test_prompt_parser_eval.py
```

### Testing

Run the test suite with:
```bash
python -m unittest discover -s tests
```

### MCP servers
- `src/audio_processor/mcp_server_audio.py`
- `src/topic_classification/mcp_topic_classification.py`
- `src/summarizer/mcp_summarizer.py`
- `src/communicator/mcp_communicator.py`

### Examples
- `tests/examples/agent_example.py`
- `tests/examples/audio_processing_examples.py`
- `tests/examples/summarizer_examples.py`
- `tests/examples/topic_classification_examples.py`
- `tests/examples/prompt_parser_example.py`

---

## Roadmap

### Near-term
- More robust multilingual workflows (transcription + summarization)
- Additional communicator backends (Discord, Teams, Telegram)
- Stronger observability (structured logs, tracing, evaluation dashboards)

### Longer-term
- Managed deployments (containerized MCP services)
- Plugin ecosystem for third-party MCP servers
- Enterprise add-ons (SSO, audit logging, compliance controls)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Email: `billiosifidis@gmail.com`  
Website: `v-iosifidis.com`