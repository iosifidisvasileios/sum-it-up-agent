# ADR: MCP-First Microservices Architecture

- Status: Accepted
- Date: 2026-02-15
- Deciders: Project maintainers
- Tags: architecture, mcp, microservices

## Context

Sum-It-Up Agent needs independent services for audio processing, classification, summarization, and communication. Requirements: independent evolution, technology diversity, clear boundaries, and scalability.

## Decision

Use Model Context Protocol for inter-service communication with four core services:
- Audio Processor (Port 9001)
- Topic Classification (Port 9002) 
- Summarizer (Port 9000)
- Communicator (Port 9003)

## Options Considered

### Monolithic
- **Pros**: Simple deployment, no network overhead
- **Cons**: Tight coupling, hard to scale components

### Traditional Microservices (HTTP/gRPC)
- **Pros**: Independent deployment, technology diversity
- **Cons**: Complex API contracts, more boilerplate

### MCP-First (chosen)
- **Pros**: Natural for LLM services, standardized interfaces, clean boundaries
- **Cons**: MCP ecosystem maturity, operational complexity

## Consequences

**Positive**: Modular development, independent scaling, clean testing, future extensibility

**Negative**: Multiple processes to manage, network latency, distributed debugging

## Implementation

- Centralized lifecycle management via orchestrator
- Port range 9000-9099 reserved for MCP services
- Environment scoping per service
- Health checking and graceful error handling
- Services designed for horizontal scaling (stateless, independent instances)
