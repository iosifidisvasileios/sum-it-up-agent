"""
Main orchestrator for the audio processing pipeline.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import Client

from .models import (
    AgentConfig,
    CommunicationChannel,
    FileValidationResult,
    PipelineResult,
    PipelineStep,
    UserIntent
)
from .prompt_parser import PromptParser
from sum_it_up_agent.observability.logger import (
    bind_request_id,
    get_logger,
    new_request_id,
)


class AudioProcessingAgent:
    """Main agent that orchestrates the entire audio processing pipeline."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Initialize prompt parser with configured LLM provider
        self.prompt_parser = PromptParser(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            prompt_limit= self.config.prompt_limit
        )
        self.logger = get_logger("sum_it_up_agent.agent.orchestrator")
        
        # MCP client sessions
        self.audio_processor_client: Optional[Client] = None
        self.topic_classifier_client: Optional[Client] = None
        self.summarizer_client: Optional[Client] = None
        self.communicator_client: Optional[Client] = None
        
        self._temp_files: List[str] = []

    async def initialize(self):
        """Initialize MCP client connections."""
        try:
            self.logger.info("Initializing MCP clients...")
            
            # Initialize audio processor client
            self.audio_processor_client = Client(self.config.audio_processor_mcp_url)
            
            # Initialize topic classifier client
            self.topic_classifier_client = Client(self.config.topic_classifier_mcp_url)
            
            # Initialize summarizer client
            self.summarizer_client = Client(self.config.summarizer_mcp_url)
            
            # Initialize communicator client
            self.communicator_client = Client(self.config.communicator_mcp_url)
            
            self.logger.info("All MCP clients initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP clients: {e}")
            raise
    
    async def process_request(
        self,
        audio_file_path: str,
        user_prompt: str
    ) -> PipelineResult:
        """
        Process a user request end-to-end.
        
        Args:
            audio_file_path: Path to the audio file
            user_prompt: User's prompt describing what they want
            
        Returns:
            PipelineResult with all execution details
        """
        start_time = time.time()
        correlation_id = new_request_id()
 
        try:
            with bind_request_id(correlation_id):
                # Parse user intent
                self.logger.info("Parsing user prompt...")
                user_intent = await self.prompt_parser.parse_prompt(user_prompt)
                self.logger.info(f"Parsed intent: {user_intent}")
                
                # Validate input file
                self.logger.info("Validating audio file...")
                file_validation = await self._validate_audio_file(audio_file_path)
                if not file_validation.is_valid:
                    return PipelineResult(
                        success=False,
                        user_intent=user_intent,
                        input_file=audio_file_path,
                        error_message=file_validation.error_message,
                    )
                
                # Initialize pipeline steps
                result = PipelineResult(
                    success=True,
                    user_intent=user_intent,
                    input_file=audio_file_path,
                    audio_processing=PipelineStep("audio_processing", "pending"),
                    topic_classification=PipelineStep("topic_classification", "pending"),
                    summarization=PipelineStep("summarization", "pending"),
                    communication=PipelineStep("communication", "pending"),
                )

                # Step 1: Audio Processing
                self.logger.info(result)
                if not await self._process_audio(result, correlation_id=correlation_id):
                    return result

                # Step 2: Topic Classification (optional)
                if user_intent.wants_summary:
                    await self._classify_topics(result, correlation_id=correlation_id)

                # Step 3: Summarization
                if user_intent.wants_summary:
                    if not await self._generate_summary(result, correlation_id=correlation_id):
                        return result

                # Step 4: Communication
                if user_intent.communication_channels:
                    await self._handle_communication(result, correlation_id=correlation_id)
                
                # Calculate total duration
                result.total_duration = time.time() - start_time
                
                self.logger.info(f"Pipeline completed successfully in {result.total_duration:.2f}s")
                return result
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                user_intent=user_intent if 'user_intent' in locals() else UserIntent(),
                input_file=audio_file_path,
                error_message=str(e),
                total_duration=time.time() - start_time,
            )
    
    async def _validate_audio_file(self, file_path: str) -> FileValidationResult:
        """Validate the audio file."""
        path = Path(file_path).expanduser().resolve()
        
        # Check if file exists
        if not path.exists():
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_size_mb=0,
                file_format="",
                exists=False,
                error_message=f"File not found: {file_path}"
            )
        
        # Check file format
        file_format = path.suffix.lower()
        if file_format not in self.config.allowed_audio_formats:
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_size_mb=0,
                file_format=file_format,
                exists=True,
                error_message=f"Unsupported file format: {file_format}"
            )
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_size_mb=file_size_mb,
                file_format=file_format,
                exists=True,
                error_message=f"File too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
            )
        
        warnings = []
        if file_size_mb > 100:
            warnings.append(f"Large file detected ({file_size_mb:.1f}MB). Processing may take time.")
        
        return FileValidationResult(
            is_valid=True,
            file_path=str(path),
            file_size_mb=file_size_mb,
            file_format=file_format,
            exists=True,
            warnings=warnings
        )
    
    async def _process_audio(self, result: PipelineResult, *, correlation_id: str) -> bool:
        """Process audio file through the audio processor MCP."""
        step = result.audio_processing
        step.status = "running"
        step.started_at = time.time()
        
        try:
            self.logger.info("Starting audio processing...")

            # Call audio processor MCP
            async with self.audio_processor_client as client:
                response = await client.call_tool(
                    "process_audio_file",
                    {
                        "audio_path": result.input_file,
                        "preset": self.config.preset_audio,
                        "output_format": self.config.output_format,
                        "output_dir": self.config.output_dir,
                        "save_to_file": True,
                        "uuid": correlation_id,
                    }
                )

            step.result = response
            step.status = "completed"
            step.completed_at = time.time()
            # Store transcription file path
            if response and hasattr(response, 'structured_content') and response.structured_content:
                if "saved_path" in response.structured_content:
                    result.transcription_file = response.structured_content["saved_path"]
                    # self.logger.info(f"Transcription saved to: {result.transcription_file}")

            self.logger.info("Audio processing completed successfully")
            return True
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.error(f"Audio processing failed: {e}")
            return False
    
    async def _classify_topics(self, result: PipelineResult, *, correlation_id: str):
        """Classify topics in the transcription."""
        step = result.topic_classification
        step.status = "running"
        step.started_at = time.time()
        
        try:
            if not result.transcription_file:
                step.status = "skipped"
                step.completed_at = time.time()
                return
            
            self.logger.info("Starting topic classification...")
            
            async with self.topic_classifier_client as client:
                response = await client.call_tool(
                    "classify_conversation_json",
                    {
                        "file_path": result.transcription_file,
                        "preset": self.config.preset_topic_classifier,
                        "export_dir": self.config.output_dir,
                        "uuid": correlation_id,
                    }
                )

            if response and hasattr(response, 'structured_content') and response.structured_content:
                if "predicted_topic" in response.structured_content:
                    result.user_intent.meeting_type = response.structured_content["predicted_topic"]
                    self.logger.info(f"Predicted topic: {result.user_intent.meeting_type}")

            step.result = response
            step.status = "completed"
            step.completed_at = time.time()
            self.logger.info("Topic classification completed")
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.warning(f"Topic classification failed: {e}")
            # Don't fail the pipeline for topic classification
    
    async def _generate_summary(self, result: PipelineResult, *, correlation_id: str) -> bool:
        """Generate summary from transcription."""
        step = result.summarization
        step.status = "running"
        step.started_at = time.time()
        
        try:
            if not result.transcription_file:
                step.status = "failed"
                step.error = "No transcription file available"
                step.completed_at = time.time()
                return False
            
            self.logger.info("Starting summarization...")

            user_preferences = ["I want the output to be " + i.value for i in result.user_intent.summary_types]
            user_preferences.extend(result.user_intent.custom_instructions)
            # Determine meeting type
            async with self.summarizer_client as client:
                response = await client.call_tool(
                    "summarize",
                    {
                        "file_path": result.transcription_file,
                        "meeting_type": result.user_intent.meeting_type,
                        "preset": self.config.preset_summarizer,
                        "user_preferences": user_preferences,
                        "output_dir": self.config.output_dir,
                        "uuid": correlation_id,
                    }
                )

            step.result = response
            step.status = "completed"
            step.completed_at = time.time()

            if response and hasattr(response, 'structured_content') and response.structured_content:
                output_path = None
                # Check if output_path is directly in structured_content
                if "output_path" in response.structured_content:
                    output_path = response.structured_content["output_path"]
                # Check if output_path is in result field
                elif "result" in response.structured_content and "output_path" in response.structured_content["result"]:
                    output_path = response.structured_content["result"]["output_path"]

                if output_path:
                    result.summary_file = output_path
                    self.logger.info(f"Summary file path: {result.summary_file}")

            self.logger.info("Summarization completed successfully")
            return True
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.error(f"Summarization failed: {e}")
            return False
    
    async def _handle_communication(self, result: PipelineResult, *, correlation_id: str):
        """Handle communication (email, slack, etc.)."""
        step = result.communication
        step.status = "running"
        step.started_at = time.time()
        
        try:
            if not result.summary_file:
                step.status = "skipped"
                step.completed_at = time.time()
                return
            
            self.logger.info("Starting communication...")
            
            communication_results = []
            
            for channel in result.user_intent.communication_channels:
                try:
                    if channel == CommunicationChannel.EMAIL:
                        email_result = await self._send_email(result, correlation_id=correlation_id)
                        communication_results.append(email_result)
                    elif channel == CommunicationChannel.SLACK:
                        slack_result = await self._send_slack(result, correlation_id=correlation_id)
                        communication_results.append(slack_result)
                    elif channel == CommunicationChannel.PDF:
                        pdf_result = await self._export_pdf(result, correlation_id=correlation_id)
                        communication_results.append(pdf_result)
                    # Add other channels as they're implemented
                    
                except Exception as e:
                    self.logger.warning(f"Failed to send via {channel}: {e}")
                    communication_results.append({
                        "channel": channel.value,
                        "success": False,
                        "error": str(e)
                    })
            
            step.result = communication_results
            result.communication_results = communication_results
            step.status = "completed"
            step.completed_at = time.time()
            
            self.logger.info("Communication completed")
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.error(f"Communication failed: {e}")
    
    async def _send_email(self, result: PipelineResult, *, correlation_id: str) -> Dict[str, Any]:
        """Send summary via email."""
        if not result.user_intent.recipients:
            raise ValueError("No recipients specified for email")

        # TODO: ADD SUBSJECT GENERATOR MCP SERVER (IF EMAIL)
        subject = result.user_intent.meeting_type or self.config.default_email_subject

        for recipient in result.user_intent.recipients:
            if "@" not  in recipient:
                self.logger.info(f"Invalid recipient: {recipient}")
                continue

            async with self.communicator_client as client:
                await client.call_tool(
                    "send_summary_email",
                    {
                        "recipient": recipient,
                        "subject": subject,
                        "summary_json_path": result.summary_file,
                        "uuid": correlation_id,
                    }
                )
        
        return {
            "channel": "email",
            "success": True,
        }
    
    async def _export_pdf(self, result: PipelineResult, *, correlation_id: str) -> Dict[str, Any]:
        """Export summary as PDF."""
        if not result.summary_file:
            raise ValueError("No summary file available for PDF export")

        subject = result.user_intent.meeting_type or self.config.default_email_subject

        async with self.communicator_client as client:
            response = await client.call_tool(
                "export_summary_pdf",
                {
                    "subject": subject,
                    "summary_json_path": result.summary_file,
                    "uuid": correlation_id,
                }
            )
            # response contains output_path; return it
            return {
                "channel": "pdf",
                "output_path": response.structured_content.get("output_path"),
                "success": True,
            }
    
    async def _send_slack(self, result: PipelineResult, *, correlation_id: str) -> Dict[str, Any]:
        """Send summary via Slack webhook."""
        if not result.summary_file:
            raise ValueError("No summary file available for Slack sending")

        subject = result.user_intent.meeting_type or self.config.default_email_subject

        async with self.communicator_client as client:
            response = await client.call_tool(
                "send_summary_slack",
                {
                    "subject": subject,
                    "summary_json_path": result.summary_file,
                    "uuid": correlation_id,
                }
            )
            
            return {
                "channel": "slack",
                "success": response.structured_content.get("success", False),
                "response": response.structured_content
            }
    
    async def cleanup(self):
        """Cleanup resources and temporary files."""
        self.logger.info("Cleaning up resources...")
        
        # Close MCP clients
        clients = [
            self.audio_processor_client,
            self.topic_classifier_client,
            self.summarizer_client,
            self.communicator_client
        ]
        
        for client in clients:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    self.logger.warning(f"Error closing client: {e}")
        
        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Error removing temp file {temp_file}: {e}")
        
        self.logger.info("Cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
