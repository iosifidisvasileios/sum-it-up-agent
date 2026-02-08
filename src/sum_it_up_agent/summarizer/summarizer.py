import json
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# LLM Provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .interfaces import (
    ISummarizer,
    SummarizationRequest,
    SummarizationResult,
    SummarizationConfig,
    SummarizationStatus,
    LLMProvider
)


class Summarizer(ISummarizer):
    """Production-grade meeting summarizer using LLMs and prompt templates."""
    
    def __init__(self, config: SummarizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._executor = ThreadPoolExecutor(max_workers=config.concurrent_requests)
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize LLM client based on provider."""
        try:
            if self.config.llm_provider == LLMProvider.OPENAI:
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI library not installed")
                
                client_kwargs = {"api_key": self.config.api_key}
                if self.config.api_base:
                    client_kwargs["base_url"] = self.config.api_base
                
                self._client = openai.OpenAI(**client_kwargs)
                self.logger.info("OpenAI client initialized")
            
            elif self.config.llm_provider == LLMProvider.ANTHROPIC:
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic library not installed")
                
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
                self.logger.info("Anthropic client initialized")
            
            elif self.config.llm_provider == LLMProvider.AZURE_OPENAI:
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI library not installed")
                
                self._client = openai.AzureOpenAI(
                    api_key=self.config.api_key,
                    azure_endpoint=self.config.api_base,
                    api_version=self.config.azure_api_version or "2024-02-15-preview"
                )
                self.logger.info("Azure OpenAI client initialized")
            
            elif self.config.llm_provider == LLMProvider.HUGGINGFACE:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("Requests library not installed")
                
                self._client = {
                    "base_url": self.config.api_base or "https://api-inference.huggingface.co",
                    "model": self.config.huggingface_model or "mistralai/Mixtral-8x7B-Instruct-v0.1"
                }
                self.logger.info("HuggingFace client initialized")
            
            elif self.config.llm_provider == LLMProvider.OLLAMA:
                if not REQUESTS_AVAILABLE:
                    raise ImportError("Requests library not installed")
                
                self._client = {
                    "base_url": self.config.ollama_host or "http://localhost:11434",
                    "model": self.config.model_name
                }
                self.logger.info("Ollama client initialized")
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def summarize_transcription(self, request: SummarizationRequest) -> SummarizationResult:
        """Summarize a transcription using appropriate prompt template."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting summarization for {request.file_path} (ID: {request_id})")
            
            # Validate request
            if not self.validate_request(request):
                raise ValueError("Invalid request")
            
            # Get prompt template
            template = self._get_prompt_template(request.meeting_type)
            if not template:
                raise ValueError(f"Unsupported meeting type: {request.meeting_type}")
            
            # Build transcript text
            transcript_text = request.get_transcript_text()
            
            # Generate prompt
            prompt = template.render(transcript_text)
            
            # Call LLM
            summary_data = self._call_llm(prompt)
            
            # Validate JSON output if required
            if self.config.validate_json_output:
                summary_data = self._validate_json_output(summary_data)
            
            processing_time = time.time() - start_time
            
            result = SummarizationResult(
                request_id=request_id,
                file_path=request.file_path,
                meeting_type=request.meeting_type,
                status=SummarizationStatus.COMPLETED,
                summary_data=summary_data,
                processing_time=processing_time,
                llm_provider=self.config.llm_provider,
                model_name=self.config.model_name,
                created_at=datetime.now().isoformat()
            )
            
            self.logger.info(f"Summarization completed in {processing_time:.2f}s")
            return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(f"Summarization failed: {error_message}")
            
            return SummarizationResult(
                request_id=request_id,
                file_path=request.file_path,
                meeting_type=request.meeting_type,
                status=SummarizationStatus.FAILED,
                error_message=error_message,
                processing_time=processing_time,
                llm_provider=self.config.llm_provider,
                model_name=self.config.model_name,
                created_at=datetime.now().isoformat()
            )

    def get_supported_meeting_types(self) -> List[str]:
        """Get list of supported meeting types."""
        try:
            from sum_it_up_agent.templates.prompts import PromptTemplateFactory
            return PromptTemplateFactory.available()
        except ImportError:
            self.logger.warning("Templates package not available")
            return []
    
    def validate_request(self, request: SummarizationRequest) -> bool:
        """Validate summarization request."""
        try:
            # Check basic requirements
            if not request.file_path:
                return False
            
            if not request.meeting_type:
                return False
            
            if not request.segments:
                return False
            
            # Check if meeting type is supported
            if request.meeting_type not in self.get_supported_meeting_types():
                return False
            
            # Check transcript length
            transcript_text = request.get_transcript_text()
            if len(transcript_text.strip()) == 0:
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            return False

    def _get_prompt_template(self, meeting_type: str):
        """Get prompt template for meeting type."""
        try:
            from sum_it_up_agent.templates.prompts import PromptTemplateFactory
            return PromptTemplateFactory.create(meeting_type)
        except ImportError:
            self.logger.error("Templates package not available")
            return None
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with prompt and return response."""
        try:
            if self.config.llm_provider == LLMProvider.OPENAI:
                return self._call_openai(prompt)
            
            elif self.config.llm_provider == LLMProvider.ANTHROPIC:
                return self._call_anthropic(prompt)
            
            elif self.config.llm_provider == LLMProvider.AZURE_OPENAI:
                return self._call_azure_openai(prompt)
            
            elif self.config.llm_provider == LLMProvider.HUGGINGFACE:
                return self._call_huggingface(prompt)
            
            elif self.config.llm_provider == LLMProvider.OLLAMA:
                return self._call_ollama(prompt)
            
            else:
                raise ValueError(f"Unsupported provider: {self.config.llm_provider}")
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides structured JSON responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from OpenAI response")
    
    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens or 4000,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.content[0].text
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from Anthropic response")
    
    def _call_azure_openai(self, prompt: str) -> Dict[str, Any]:
        """Call Azure OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.azure_deployment or self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides structured JSON responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        content = response.choices[0].message.content
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from Azure OpenAI response")
    
    def _call_huggingface(self, prompt: str) -> Dict[str, Any]:
        """Call HuggingFace API."""
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens or 4000,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"{self._client['base_url']}/models/{self._client['model']}",
            headers=headers,
            json=data,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            content = result[0].get("generated_text", "")
        else:
            content = str(result)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from HuggingFace response")
    
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API."""
        data = {
            "model": self._client["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens or 8000
            }
        }
        
        response = requests.post(
            f"{self._client['base_url']}/api/generate",
            json=data,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        del result["context"]

        try:
            return result
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON from Ollama response")
    
    def _validate_json_output(self, data: Any) -> Dict[str, Any]:
        """Validate and clean JSON output."""
        if isinstance(data, dict):
            return data
        else:
            raise ValueError("Response is not valid JSON")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # Close client connections if needed
            if hasattr(self._client, 'close'):
                self._client.close()
            
            self.logger.info("Cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
