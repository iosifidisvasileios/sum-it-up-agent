# ADR: Observability and Monitoring

- Status: Accepted
- Date: 2026-02-15
- Deciders: Project maintainers
- Tags: observability, monitoring, logging

## Context

Distributed MCP services need request correlation, performance tracking, and error debugging across the pipeline.

## Decision

Implement structured logging with correlation IDs and pipeline step timing.

### Core Components
- **Structured Logging**: Centralized logger with request correlation
- **Timing Metrics**: Track duration per pipeline step
- **Health Monitoring**: MCP server health checks

## Options Considered

### Basic Print Statements
- **Pros**: Simple
- **Cons**: No structure or correlation

### External APM (DataDog/New Relic)
- **Pros**: Full observability suite
- **Cons**: Cost, complexity, overkill

### Custom Structured Logging (chosen)
- **Pros**: Control, no dependencies, correlation, extensible
- **Cons**: Manual instrumentation, log volume

## Consequences

**Positive**: Request correlation, performance visibility, debugging efficiency

**Negative**: Manual instrumentation, analysis effort required

## Implementation

- Correlation ID propagation across services
- Pipeline step timing in PipelineResult
- Environment-configurable logging (console + optional file)
- Structured log format: `%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s`

## Follow-ups

- Add metrics collection (Prometheus/OpenTelemetry)
- Implement distributed tracing
- Create log analysis dashboards
- Add alerting for errors and performance
