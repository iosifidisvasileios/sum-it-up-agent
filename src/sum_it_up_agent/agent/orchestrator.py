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


class AudioProcessingAgent:
    """Main agent that orchestrates the entire audio processing pipeline."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Initialize prompt parser with configured LLM provider
        self.prompt_parser = PromptParser(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url
        )
        self.logger = self._setup_logging()
        
        # MCP client sessions
        self.audio_processor_client: Optional[Client] = None
        self.topic_classifier_client: Optional[Client] = None
        self.summarizer_client: Optional[Client] = None
        self.communicator_client: Optional[Client] = None
        
        self._temp_files: List[str] = []

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
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
        
        try:
            # Parse user intent
            self.logger.info("Parsing user prompt...")
            user_intent = self.prompt_parser.parse_prompt(user_prompt)
            self.logger.info(f"Parsed intent: {user_intent}")
            
            # Validate input file
            self.logger.info("Validating audio file...")
            file_validation = await self._validate_audio_file(audio_file_path)
            if not file_validation.is_valid:
                return PipelineResult(
                    success=False,
                    user_intent=user_intent,
                    input_file=audio_file_path,
                    error_message=file_validation.error_message
                )
            
            # Initialize pipeline steps
            result = PipelineResult(
                success=True,
                user_intent=user_intent,
                input_file=audio_file_path,
                audio_processing=PipelineStep("audio_processing", "pending"),
                topic_classification=PipelineStep("topic_classification", "pending"),
                summarization=PipelineStep("summarization", "pending"),
                communication=PipelineStep("communication", "pending")
            )
            
            # Step 1: Audio Processing
            if not await self._process_audio(result):
                return result
            
            # Step 2: Topic Classification (optional)
            if user_intent.wants_summary:
                await self._classify_topics(result)
            
            # Step 3: Summarization
            if user_intent.wants_summary:
                if not await self._generate_summary(result):
                    return result
            
            # Step 4: Communication
            if user_intent.communication_channels:
                await self._handle_communication(result)
            
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
                total_duration=time.time() - start_time
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
    
    async def _process_audio(self, result: PipelineResult) -> bool:
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
                        "preset": result.user_intent.audio_preset,
                        "output_format": result.user_intent.output_format,
                        "save_to_file": True
                    }
                )
            
            step.result = response
            step.status = "completed"
            step.completed_at = time.time()
            
            # Store transcription file path
            if response and "saved_path" in response:
                result.transcription_file = response["saved_path"]
            
            self.logger.info("Audio processing completed successfully")
            return True
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.error(f"Audio processing failed: {e}")
            return False
    
    async def _classify_topics(self, result: PipelineResult):
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
                        "preset": "standard"
                    }
                )
            
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
    
    async def _generate_summary(self, result: PipelineResult) -> bool:
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
            
            # Determine meeting type
            meeting_type = result.user_intent.meeting_type or self.config.default_meeting_type
            
            async with self.summarizer_client as client:
                response = await client.call_tool(
                    "summarize",
                    {
                        "file_path": result.transcription_file,
                        "meeting_type": meeting_type,
                        "preset": result.user_intent.summarizer_preset,
                        "output_dir": os.path.dirname(result.transcription_file)
                    }
                )
            
            step.result = response
            step.status = "completed"
            step.completed_at = time.time()
            
            # Store summary file path
            if response and "result" in response and "output_path" in response["result"]:
                result.summary_file = response["result"]["output_path"]
            
            self.logger.info("Summarization completed successfully")
            return True
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = time.time()
            self.logger.error(f"Summarization failed: {e}")
            return False
    
    async def _handle_communication(self, result: PipelineResult):
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
                        email_result = await self._send_email(result)
                        communication_results.append(email_result)
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
    
    async def _send_email(self, result: PipelineResult) -> Dict[str, Any]:
        """Send summary via email."""
        if not result.user_intent.recipients:
            raise ValueError("No recipients specified for email")
        
        subject = result.user_intent.subject or self.config.default_email_subject
        
        response = await self.communicator_client.call_tool(
            "send_summary_email",
            {
                "recipient": result.user_intent.recipients[0],  # TODO: Support multiple recipients
                "subject": subject,
                "summary_json_path": result.summary_file
            }
        )
        
        return {
            "channel": "email",
            "success": True,
            "result": response.content
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
