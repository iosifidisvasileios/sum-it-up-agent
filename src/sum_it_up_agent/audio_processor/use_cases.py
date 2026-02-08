import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from sum_it_up_agent.audio_processor import  IAudioProcessor, TranscriptionSegment
from sum_it_up_agent.audio_processor import ProcessorType, AudioProcessorFactory


class AudioProcessingUseCase:
    """Use case class for audio processing business logic."""
    
    def __init__(self, processor: Optional[IAudioProcessor] = None):
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    def process_audio_file(
        self,
        audio_path: str,
        output_format: str = "json",
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> List[TranscriptionSegment]:
        """Process an audio file and return transcription segments."""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.logger.info(f"Processing audio file: {audio_path}")
            
            # Process audio
            segments = self.processor.process_audio(str(audio_path))
            
            # Save results if requested
            if save_to_file:
                self._save_results(segments, audio_path, output_format, output_dir)
            
            self.logger.info(f"Processing completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to process audio file: {e}")
            raise
    
    def process_multiple_files(
        self,
        audio_paths: List[str],
        output_format: str = "json",
        output_dir: Optional[str] = None
    ) -> Dict[str, List[TranscriptionSegment]]:
        """Process multiple audio files."""
        results = {}
        
        for audio_path in audio_paths:
            try:
                self.logger.info(f"Processing file: {audio_path}")
                segments = self.process_audio_file(
                    audio_path,
                    output_format,
                    save_to_file=True,
                    output_dir=output_dir
                )
                results[audio_path] = segments
                
            except Exception as e:
                self.logger.error(f"Failed to process {audio_path}: {e}")
                results[audio_path] = []
        
        return results
    
    def get_transcription_summary(self, segments: List[TranscriptionSegment]) -> Dict[str, Any]:
        """Generate summary statistics for transcription."""
        if not segments:
            return {"total_segments": 0, "total_duration": 0, "speakers": [], "word_count": 0}
        
        speakers = set(segment.speaker for segment in segments)
        total_duration = max(segment.end_time for segment in segments)
        word_count = sum(len(segment.text.split()) for segment in segments)
        
        speaker_stats = {}
        for speaker in speakers:
            speaker_segments = [s for s in segments if s.speaker == speaker]
            speaker_duration = sum(s.end_time - s.start_time for s in speaker_segments)
            speaker_words = sum(len(s.text.split()) for s in speaker_segments)
            
            speaker_stats[speaker] = {
                "segment_count": len(speaker_segments),
                "duration": speaker_duration,
                "word_count": speaker_words,
                "avg_segment_length": speaker_duration / len(speaker_segments) if speaker_segments else 0
            }
        
        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "speakers": list(speakers),
            "word_count": word_count,
            "speaker_stats": speaker_stats,
            "avg_words_per_segment": word_count / len(segments) if segments else 0
        }
    
    def export_transcription(
        self,
        segments: List[TranscriptionSegment],
        output_path: str,
        format_type: str = "json"
    ) -> None:
        """Export transcription to different formats."""
        output_path = Path(output_path)
        
        if format_type.lower() == "json":
            self._export_json(segments, output_path)
        elif format_type.lower() == "txt":
            self._export_txt(segments, output_path)
        elif format_type.lower() == "srt":
            self._export_srt(segments, output_path)
        elif format_type.lower() == "csv":
            self._export_csv(segments, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Transcription exported to: {output_path}")
    
    def _save_results(
        self,
        segments: List[TranscriptionSegment],
        original_path: Path,
        output_format: str,
        output_dir: Optional[str]
    ) -> None:
        """Save processing results to file."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{original_path.stem}_transcription.{output_format}"
        else:
            output_path = original_path.parent / f"{original_path.stem}_transcription.{output_format}"
        
        self.export_transcription(segments, str(output_path), output_format)
    
    def _export_json(self, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Export as JSON format."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_segments": len(segments),
                "summary": self.get_transcription_summary(segments)
            },
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.end_time - seg.start_time,
                    "speaker": seg.speaker,
                    "text": seg.text
                }
                for seg in segments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_txt(self, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Export as plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"[{segment.start_time:.2f}-{segment.end_time:.2f}] {segment.speaker}: {segment.text}\n")
    
    def _export_srt(self, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Export as SRT subtitle format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_srt_time(segment.start_time)
                end_time = self._format_srt_time(segment.end_time)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.speaker}: {segment.text}\n\n")
    
    def _export_csv(self, segments: List[TranscriptionSegment], output_path: Path) -> None:
        """Export as CSV format."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Start Time', 'End Time', 'Duration', 'Speaker', 'Text'])
            
            for segment in segments:
                writer.writerow([
                    segment.start_time,
                    segment.end_time,
                    segment.end_time - segment.start_time,
                    segment.speaker,
                    segment.text
                ])
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT subtitles."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @classmethod
    def create_with_preset(
        cls,
        processor_type: ProcessorType = ProcessorType.STANDARD,
        huggingface_token: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> 'AudioProcessingUseCase':
        """Create use case with preset processor."""
        processor = AudioProcessorFactory.create_processor(
            processor_type=processor_type,
            huggingface_token=huggingface_token,
            config_overrides=config_overrides
        )
        return cls(processor)
    
    @classmethod
    def create_with_custom_processor(cls, config) -> 'AudioProcessingUseCase':
        """Create use case with custom processor."""
        processor = AudioProcessorFactory.create_custom_processor(config)
        return cls(processor)
