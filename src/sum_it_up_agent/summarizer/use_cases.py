import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import csv

from .interfaces import (
    ISummarizer,
    SummarizationRequest,
    SummarizationResult,
    TranscriptionSegment,
)
from .factory import SummarizerFactory, SummarizerType


class SummarizationUseCase:
    """Use case class for meeting summarization business logic."""
    
    def __init__(self, summarizer: Optional[ISummarizer] = None):
        self.summarizer = summarizer
        self.logger = logging.getLogger(__name__)
    
    def summarize_transcription_file(
        self,
        file_path: str,
        meeting_type: str,
        user_preferences: list = None,
        output_dir: Optional[str] = None
    ) -> SummarizationResult:
        """Summarize a transcription file."""
        try:
            self.logger.info(f"Processing transcription file: {file_path}")
            
            # Load transcription data
            segments = self._load_transcription_segments(file_path)
            metadata = self._extract_metadata(file_path)
            
            # Create request
            request = SummarizationRequest(
                file_path=file_path,
                meeting_type=meeting_type,
                segments=segments,
                metadata=metadata,
                user_preferences=user_preferences
            )
            
            # Summarize
            result = self.summarizer.summarize_transcription(request)
            
            # Save result if output directory specified
            if output_dir and result.is_successful():
                self._save_result(result, output_dir)
            
            self.logger.info(f"Summarization completed: {result.status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to summarize transcription file: {e}")
            raise


    def analyze_summarization_results(self, results: List[SummarizationResult]) -> Dict[str, Any]:
        """Analyze summarization results and generate statistics."""
        if not results:
            return {"total_files": 0, "error": "No results to analyze"}
        
        # Basic statistics
        total_files = len(results)
        successful_files = sum(1 for r in results if r.is_successful())
        failed_files = total_files - successful_files
        
        total_processing_time = sum(r.processing_time or 0 for r in results)
        
        # Meeting type distribution
        meeting_type_counts = {}
        for result in results:
            meeting_type = result.meeting_type
            meeting_type_counts[meeting_type] = meeting_type_counts.get(meeting_type, 0) + 1
        
        # Provider distribution
        provider_counts = {}
        for result in results:
            provider = result.llm_provider.value if result.llm_provider else "unknown"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        # Performance metrics
        avg_processing_time = total_processing_time / total_files if total_files > 0 else 0
        
        # Extract summary statistics
        summary_lengths = []
        executive_summary_lengths = []
        
        for result in results:
            if result.is_successful() and result.summary_data:
                summary_json = json.dumps(result.summary_data)
                summary_lengths.append(len(summary_json))
                
                executive_summary = result.summary_data.get("executive_summary", "")
                if executive_summary:
                    executive_summary_lengths.append(len(executive_summary))
        
        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": successful_files / total_files if total_files > 0 else 0,
            "total_processing_time": total_processing_time,
            "avg_processing_time": avg_processing_time,
            "meeting_type_distribution": meeting_type_counts,
            "provider_distribution": provider_counts,
            "avg_summary_length": sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0,
            "avg_executive_summary_length": sum(executive_summary_lengths) / len(executive_summary_lengths) if executive_summary_lengths else 0,
            "most_common_meeting_type": max(meeting_type_counts.items(), key=lambda x: x[1])[0] if meeting_type_counts else None
        }
    
    def export_results(
        self,
        results: List[SummarizationResult],
        output_path: str,
        format_type: str = "json",
        include_analysis: bool = True,
        include_raw_summaries: bool = True
    ) -> None:
        """Export summarization results to different formats."""
        output_path = Path(output_path)
        
        if format_type.lower() == "json":
            self._export_json(results, output_path, include_analysis, include_raw_summaries)
        elif format_type.lower() == "csv":
            self._export_csv(results, output_path)
        elif format_type.lower() == "txt":
            self._export_txt(results, output_path, include_analysis)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Results exported to: {output_path}")
    
    def _load_transcription_segments(self, file_path: str) -> List[TranscriptionSegment]:
        """Load transcription segments from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self._extract_segments_from_audio_output(data)
        
        except Exception as e:
            self.logger.error(f"Failed to load transcription segments: {e}")
            raise
    
    def _extract_segments_from_audio_output(self, data: Dict[str, Any]) -> List[TranscriptionSegment]:
        """Extract segments from audio processor output format."""
        segments = []
        
        # Handle different audio processor output formats
        if "segments" in data:
            # Direct segments format
            for segment_data in data["segments"]:
                segments.append(TranscriptionSegment(
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", 0.0),
                    speaker=segment_data.get("speaker", "UNKNOWN"),
                    text=segment_data.get("text", "")
                ))
        
        elif "metadata" in data and "segments" in data["metadata"]:
            # Nested metadata format
            for segment_data in data["metadata"]["segments"]:
                segments.append(TranscriptionSegment(
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", 0.0),
                    speaker=segment_data.get("speaker", "UNKNOWN"),
                    text=segment_data.get("text", "")
                ))
        
        else:
            # Try to find segments anywhere in the data
            segments_list = self._find_segments_recursive(data)
            for segment_data in segments_list:
                segments.append(TranscriptionSegment(
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", 0.0),
                    speaker=segment_data.get("speaker", "UNKNOWN"),
                    text=segment_data.get("text", "")
                ))
        
        return segments
    
    def _find_segments_recursive(self, obj: Any) -> List[Dict[str, Any]]:
        """Recursively find segments in nested data structure."""
        if isinstance(obj, dict):
            if "segments" in obj and isinstance(obj["segments"], list):
                return obj["segments"]
            for value in obj.values():
                segments = self._find_segments_recursive(value)
                if segments:
                    return segments
        elif isinstance(obj, list):
            for item in obj:
                segments = self._find_segments_recursive(item)
                if segments:
                    return segments
        return []
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file."""
        try:
            file_path_obj = Path(file_path)
            
            metadata = {
                "file_name": file_path_obj.name,
                "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else 0,
                "processed_at": datetime.now().isoformat()
            }
            
            # Try to load additional metadata from the file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "metadata" in data:
                metadata.update(data["metadata"])
            
            return metadata
        
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")
            return {"file_name": Path(file_path).name}
    
    def _save_result(self, result: SummarizationResult, output_dir: str) -> None:
        """Save summarization result to file."""
        try:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            file_name = Path(result.file_path).stem
            output_file = output_dir_path / f"{file_name}_summary.json"
            
            result.save_to_file(str(output_file))
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    def _export_json(
        self,
        results: List[SummarizationResult],
        output_path: Path,
        include_analysis: bool,
        include_raw_summaries: bool
    ) -> None:
        """Export results as JSON format."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_results": len(results),
                "successful_results": sum(1 for r in results if r.is_successful())
            }
        }
        
        if include_analysis:
            data["analysis"] = self.analyze_summarization_results(results)
        
        if include_raw_summaries:
            data["results"] = []
            for result in results:
                result_data = {
                    "request_id": result.request_id,
                    "file_path": result.file_path,
                    "meeting_type": result.meeting_type,
                    "status": result.status.value,
                    "processing_time": result.processing_time,
                    "llm_provider": result.llm_provider.value if result.llm_provider else None,
                    "model_name": result.model_name,
                    "summary_data": result.summary_data if result.is_successful() else None,
                    "error_message": result.error_message
                }
                data["results"].append(result_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, results: List[SummarizationResult], output_path: Path) -> None:
        """Export results as CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'file_path', 'meeting_type', 'status', 'processing_time',
                'llm_provider', 'model_name', 'summary_available',
                'executive_summary', 'error_message'
            ])
            
            # Data rows
            for result in results:
                executive_summary = ""
                if result.is_successful() and result.summary_data:
                    executive_summary = result.summary_data.get("executive_summary", "")
                
                writer.writerow([
                    result.file_path,
                    result.meeting_type,
                    result.status.value,
                    result.processing_time,
                    result.llm_provider.value if result.llm_provider else "",
                    result.model_name,
                    result.is_successful(),
                    executive_summary,
                    result.error_message
                ])
    
    def _export_txt(
        self,
        results: List[SummarizationResult],
        output_path: Path,
        include_analysis: bool
    ) -> None:
        """Export results as plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Meeting Summarization Results\n")
            f.write("=" * 50 + "\n\n")
            
            if include_analysis:
                analysis = self.analyze_summarization_results(results)
                f.write("Summary Analysis:\n")
                f.write(f"Total files: {analysis['total_files']}\n")
                f.write(f"Successful: {analysis['successful_files']}\n")
                f.write(f"Success rate: {analysis['success_rate']:.2%}\n")
                f.write(f"Average processing time: {analysis['avg_processing_time']:.2f}s\n")
                f.write(f"Most common meeting type: {analysis['most_common_meeting_type']}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 50 + "\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\n{i}. {Path(result.file_path).name}\n")
                f.write(f"   Meeting Type: {result.meeting_type}\n")
                f.write(f"   Status: {result.status.value}\n")
                f.write(f"   Processing Time: {result.processing_time:.2f}s\n")
                
                if result.is_successful() and result.summary_data:
                    exec_summary = result.summary_data.get("executive_summary", "")
                    if exec_summary:
                        f.write(f"   Executive Summary: {exec_summary}\n")
                elif result.error_message:
                    f.write(f"   Error: {result.error_message}\n")
    
    def get_supported_meeting_types(self) -> List[str]:
        """Get list of supported meeting types."""
        return self.summarizer.get_supported_meeting_types()
    

    @classmethod
    def create_with_preset(
        cls,
        summarizer_type: SummarizerType = SummarizerType.OPENAI_STANDARD,
        api_key: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> 'SummarizationUseCase':
        """Create use case with preset summarizer."""
        summarizer = SummarizerFactory.create_summarizer(
            summarizer_type=summarizer_type,
            api_key=api_key,
            config_overrides=config_overrides
        )
        return cls(summarizer)
    
    @classmethod
    def create_from_environment(cls) -> 'SummarizationUseCase':
        """Create use case from environment variables."""
        summarizer = SummarizerFactory.create_from_environment()
        return cls(summarizer)
    
    @classmethod
    def create_with_custom_summarizer(cls, config) -> 'SummarizationUseCase':
        """Create use case with custom summarizer."""
        summarizer = SummarizerFactory.create_custom_summarizer(config)
        return cls(summarizer)
