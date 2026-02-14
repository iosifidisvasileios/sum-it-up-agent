graph TB
    subgraph "Client Layer"
        CLI[Interactive Interface]
        APP[Python Application]
    end
    
    subgraph "Orchestration Layer"
        ORCHESTRATOR[AudioProcessingAgent]
        PARSER[PromptParser]
    end
    
    subgraph "MCP Services Layer"
        AUDIO_MCP[Audio Processor MCP<br/>Port 9001]
        TOPIC_MCP[Topic Classifier MCP<br/>Port 9002]
        SUMMARIZER_MCP[Summarizer MCP<br/>Port 9000]
        COMMUNICATOR_MCP[Communicator MCP<br/>Port 9003]
    end
    
    subgraph "Communication Backends"
        EMAIL[Email Service]
        SLACK[Slack Webhook]
        PDF[PDF Exporter]
    end
    
    CLI --> ORCHESTRATOR
    APP --> ORCHESTRATOR
    ORCHESTRATOR --> PARSER
    
    ORCHESTRATOR --> AUDIO_MCP
    ORCHESTRATOR --> TOPIC_MCP
    ORCHESTRATOR --> SUMMARIZER_MCP
    ORCHESTRATOR --> COMMUNICATOR_MCP
    
    COMMUNICATOR_MCP --> EMAIL
    COMMUNICATOR_MCP --> SLACK
    COMMUNICATOR_MCP --> PDF
    
    AUDIO_MCP --> ORCHESTRATOR
    TOPIC_MCP --> ORCHESTRATOR
    SUMMARIZER_MCP --> ORCHESTRATOR
    COMMUNICATOR_MCP --> ORCHESTRATOR