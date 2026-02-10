# Sum-It-Up Agent

The main AI agent that orchestrates the entire audio processing pipeline using MCP servers and LLM-based prompt parsing.

## Overview

The `AudioProcessingAgent` is the central component that:
- Uses advanced LLM-based parsing to understand natural language user requests
- Extracts structured intents, communication channels, and processing requirements
- Validates input audio files and orchestrates the processing pipeline using MCP servers
- Handles multiple communication channels (email, Slack, Discord, Telegram, Jira)
- Provides comprehensive error handling and logging with graceful degradation
- Adapts processing strategy based on user intent and content characteristics

## Quick Start

### Command Line Usage

```bash
# Basic usage
python -m sum_it_up_agent.agent.main audio_file.mp3 "Please summarize and send to email@example.com"

# With options
python -m sum_it_up_agent.agent.main meeting.wav \
  "Extract action items and send to team@company.com" \
  --audio-preset high_quality \
  --summarizer-preset openai_detailed \
  --verbose \
  --output-json results.json

# Advanced natural language requests
python -m sum_it_up_agent.agent.main client_call.mp3 \
  "Transcribe this client meeting, create an executive summary with key decisions and action items, \
  send to john@company.com with subject 'Client Meeting - Q4 Planning' and also post \
  action items to our Jira board"
```

### Programmatic Usage

```python
import asyncio
from sum_it_up_agent.agent import AudioProcessingAgent, AgentConfig

async def process_audio():
    # Configure with your preferred LLM provider
    config = AgentConfig(
        llm_provider="openai",  # or "anthropic", "ollama"
        llm_model="gpt-4-turbo",
        max_file_size_mb=200
    )
    
    async with AudioProcessingAgent(config) as agent:
        # Natural language request - the agent will parse and understand
        result = await agent.process_request(
            "meeting.mp3",
            "Please summarize and send action points to john@example.com"
        )
        
        if result.success:
            print(f"Success! Summary saved to: {result.summary_file}")
            print(f"Communication results: {result.communication_results}")
        else:
            print(f"Failed: {result.error_message}")

asyncio.run(process_audio())
```

## Features

### Advanced LLM-Based Prompt Parsing

The agent uses a sophisticated LLM-based approach for intent detection, making it truly intelligent. Instead of rigid regex patterns, it leverages configurable language models to understand natural language requests with contextual awareness.

**Supported LLM Providers:**
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Anthropic**: Claude-3-haiku, Claude-3-sonnet, Claude-3-opus
- **Ollama**: Local models like Llama3.1, Mistral, CodeLlama (no API key required)

**How it works:**
1. **LLM Analysis**: The user prompt is sent to the configured LLM with a detailed system prompt
2. **Structured Extraction**: The LLM returns a structured JSON with all detected intents
3. **Type Safety**: The response is validated and converted to strongly-typed objects
4. **Fallback Protection**: If LLM fails, basic keyword analysis ensures core functionality

**What the LLM detects:**
- **Communication channels**: email, slack, discord, telegram, jira
- **Summary types**: action_items, decisions, key_points, detailed, bullet_points, executive, standard
- **Meeting types**: business, team, project, client, interview, training, conference, general
- **Recipients**: email addresses and names from context
- **Subject lines**: explicit or inferred from content
- **Processing preferences**: transcription-only, urgency, language
- **Custom requirements**: focus areas, inclusions, exclusions
- **Custom instructions**: any remaining contextual information

**Advantages over regex:**
- **Natural language understanding**: Handles variations, context, and complex requests
- **Ambiguity resolution**: Can interpret unclear requests and make intelligent inferences
- **Scalability**: Easy to add new intent types without code changes
- **Context awareness**: Understands relationships between different parts of the prompt
- **Robustness**: Handles typos, grammatical errors, and varied phrasing
- **Provider flexibility**: Choose the best LLM for your needs and budget

### Multi-Channel Communication

The agent supports multiple communication channels with intelligent parsing:
- **Email**: SMTP with attachment support, multiple recipients
- **Slack**: Webhook integration for channel posting
- **Discord**: Bot integration for server channels
- **Telegram**: Bot API for individual and group messages
- **Jira**: REST API integration for issue creation and updates

### Intelligent Pipeline Orchestration

1. **File Validation**: Checks file existence, format, and size
2. **Intent Parsing**: LLM-based extraction of user requirements
3. **Audio Processing**: Transcribes audio using the audio processor MCP
4. **Topic Classification**: Classifies conversation topics (optional)
5. **Summarization**: Generates summary using the summarizer MCP
6. **Communication**: Sends results via specified channels
7. **Error Recovery**: Graceful handling of failures with detailed reporting

### Error Handling

- Comprehensive error reporting with step-by-step details
- Graceful degradation (e.g., continues if topic classification fails)
- Detailed logging for debugging
- Warning messages for potential issues

## Configuration

### AgentConfig

```python
config = AgentConfig(
    # File settings
    allowed_audio_formats=[".wav", ".mp3", ".m4a", ".flac"],
    max_file_size_mb=500,
    temp_dir="/tmp/sum_it_up",
    
    # LLM Provider Configuration
    llm_provider="openai",  # openai, anthropic, ollama
    llm_model="gpt-3.5-turbo",
    llm_base_url=None,  # Only needed for Ollama or custom endpoints
    
    # MCP server URLs
    audio_processor_mcp_url="stdio://audio-processor",
    topic_classifier_mcp_url="stdio://topic-classifier",
    summarizer_mcp_url="stdio://summarizer",
    communicator_mcp_url="stdio://communicator",
    
    # Processing defaults
    default_audio_preset="standard",
    default_summarizer_preset="openai_standard",
    default_meeting_type="general",
    
    # Communication settings
    default_email_subject="Meeting Summary",
    max_recipients=50
)

# Provider-specific examples:

# OpenAI Configuration
config_openai = AgentConfig(
    llm_provider="openai",
    llm_model="gpt-4-turbo"
)

# Anthropic Configuration  
config_anthropic = AgentConfig(
    llm_provider="anthropic",
    llm_model="claude-3-sonnet-20240229"
)

# Ollama Configuration (Local)
config_ollama = AgentConfig(
    llm_provider="ollama",
    llm_model="llama3.1",
    llm_base_url="http://localhost:11434"
)
```

### Environment Variables

The agent uses environment variables for configuration:

```bash
# LLM for Intent Detection (choose one)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
# Ollama doesn't need API key for local use

# Audio Processor
HUGGINGFACE_TOKEN=your_token_here
MCP_TRANSPORT_AUDIO_PROCESSOR=stdio

# Summarizer
OPENAI_API_KEY=your_openai_key_here  # Reused for summarization
ANTHROPIC_API_KEY=your_anthropic_key_here  # Reused for summarization
MCP_TRANSPORT_SUMMARIZER=stdio

# Communicator
SENDER_EMAIL_ACCOUNT=your_email@gmail.com
SENDER_EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
MCP_TRANSPORT_COMMUNICATOR=stdio

# Optional: Custom endpoints
OLLAMA_BASE_URL=http://localhost:11434  # For local Ollama
```

## Example Prompts

### Basic Examples
- "Please summarize this meeting"
- "Transcribe and send to email@example.com"
- "Extract action items and post to Slack"

### Advanced Examples
- "Please extract the conversation, summarize it with focus on decisions and action items, and send to john@company.com with subject 'Project Meeting Summary'"
- "Transcribe this client call, create a detailed summary with key points and decisions, and send to the team email list"
- "Just transcribe this interview without summarizing, save as JSON"
- "Process this training session and create bullet points of main topics, send via email with high priority"
- "Extract action items from this meeting and add them to our Jira board under the 'Project X' project"
- "Create an executive summary of this board meeting, focus on financial decisions, and email to the CFO and CEO"

The LLM-based parser can understand nuanced requests like:
- "Give me the highlights from this business meeting, focus on financial decisions, and email them to the CFO"
- "I need a quick executive summary of this project update, skip the details, just send action items to the team"
- "This is an urgent client call - extract any commitments made and send them immediately via email"
- "Post the key decisions from this retrospective to our Slack channel and create Jira tickets for action items"

## Output

### PipelineResult

The agent returns a comprehensive `PipelineResult` object containing:

```python
{
    "success": True,
    "user_intent": UserIntent(...),
    "input_file": "meeting.mp3",
    "audio_processing": PipelineStep(...),
    "topic_classification": PipelineStep(...),
    "summarization": PipelineStep(...),
    "communication": PipelineStep(...),
    "transcription_file": "/path/to/transcription.json",
    "summary_file": "/path/to/summary.json",
    "communication_results": [...],
    "total_duration": 45.2,
    "warnings": ["Large file detected, processing may take time"]
}
```

## CLI Options

```
usage: python -m sum_it_up_agent.agent.main [-h] [--config CONFIG]
                                             [--audio-preset AUDIO_PRESET]
                                             [--summarizer-preset SUMMARIZER_PRESET]
                                             [--output-format {json,txt,srt,csv}]
                                             [--save-intermediate]
                                             [--max-file-size MAX_FILE_SIZE]
                                             [--verbose] [--output-json OUTPUT_JSON]
                                             audio_file prompt

positional arguments:
  audio_file            Path to the audio file to process
  prompt                User prompt describing what to do with the audio

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (JSON)
  --audio-preset        Audio processing preset (default: standard)
  --summarizer-preset   Summarizer preset (default: openai_standard)
  --output-format       Output format for transcription (default: json)
  --save-intermediate   Save intermediate processing results
  --max-file-size       Maximum file size in MB (default: 500)
  --verbose, -v         Enable verbose logging
  --output-json         Save pipeline result to JSON file
```

## Error Scenarios

The agent handles various error scenarios gracefully:

- **File not found**: Clear error message with file path
- **Unsupported format**: Lists supported formats
- **File too large**: Shows size and limit
- **MCP server unavailable**: Continues with available services
- **Processing failures**: Detailed error messages per step
- **Communication failures**: Reports per-channel results

## Extending the Agent

### Adding New Communication Channels

1. Add channel to `CommunicationChannel` enum in `models.py`
2. Update the LLM prompt template in `prompt_parser.py` to recognize the new channel
3. Implement handler in `_handle_communication()` in `orchestrator.py`
4. Add MCP server integration if needed

### Adding New Summary Types

1. Add type to `SummaryType` enum in `models.py`
2. Update the LLM prompt template to detect new summary types
3. Update summarizer MCP to handle new types
4. Add corresponding templates in the templates module

### Custom Prompt Patterns

The LLM-based approach makes customization much easier:
- Update the system prompt in `prompt_parser.py` to handle new types of user requests
- Add new examples to the prompt template to improve recognition
- Modify the structured output schema to include new fields
- No need to maintain complex regex patterns
