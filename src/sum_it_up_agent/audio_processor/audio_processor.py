import logging
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
from faster_whisper import WhisperModel

from sum_it_up_agent.audio_processor import IAudioProcessor, AudioProcessingConfig, TranscriptionSegment, DeviceType


class AudioProcessor(IAudioProcessor):
    """Production-grade audio processor with speaker diarization and transcription."""
    
    def __init__(self, config: AudioProcessingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._pipeline = None
        self._whisper_model = None
        self._device = torch.device(config.device.value)
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize diarization pipeline and Whisper model."""
        try:
            # Initialize diarization pipeline
            if not self.config.huggingface_token:
                raise ValueError("HuggingFace token is required for diarization")
            
            self.logger.info("Initializing diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
                token=self.config.huggingface_token
            )
            self._pipeline.to(self._device)
            
            # Initialize Whisper model
            self.logger.info(f"Initializing Whisper model: {self.config.whisper_model}")
            self._whisper_model = WhisperModel(
                self.config.whisper_model,
                device=self.config.device.value,
                compute_type=self.config.compute_type
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def convert_audio_format(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Convert audio to WAV format with specified sample rate."""
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if output_path is None:
                output_path = str(input_path.with_suffix(".wav"))
            
            self.logger.info(f"Converting {input_path} to {output_path}")
            
            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_channels(1).set_frame_rate(self.config.sample_rate).set_sample_width(2)
            audio.export(output_path, format="wav")
            
            self.logger.info(f"Audio conversion completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            raise
    
    def perform_diarization(self, audio_path: str) -> object:
        """Perform speaker diarization on audio file."""
        try:
            self.logger.info(f"Starting diarization for: {audio_path}")
            
            with ProgressHook() as hook:
                output = self._pipeline(audio_path, hook=hook)
            
            self.logger.info("Diarization completed successfully")
            return output.speaker_diarization
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> List[Tuple]:
        """Transcribe audio using Whisper model."""
        try:
            self.logger.info(f"Starting transcription for: {audio_path}")
            
            segments, info = self._whisper_model.transcribe(
                audio_path,
                vad_filter=self.config.vad_filter
            )
            
            transcription_segments = []
            for segment in segments:
                transcription_segments.append((
                    float(segment.start),
                    float(segment.end),
                    segment.text.strip()
                ))
            
            self.logger.info(f"Transcription completed: {len(transcription_segments)} segments")
            return transcription_segments
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _pick_speaker(self, diarization, start: float, end: float) -> str:
        """Return speaker with max overlap inside [start, end]."""
        try:
            clip = diarization.crop(Segment(start, end), mode="intersection")
            if clip is None:
                return "UNKNOWN"
            
            totals = defaultdict(float)
            for seg, _, spk in clip.itertracks(yield_label=True):
                totals[spk] += seg.duration
            
            return max(totals, key=totals.get) if totals else "UNKNOWN"
            
        except Exception as e:
            self.logger.warning(f"Failed to pick speaker for segment [{start}-{end}]: {e}")
            return "UNKNOWN"

    def merge_segments_by_speaker(self, segments: List[Tuple]) -> List[TranscriptionSegment]:
        """Merge adjacent segments when the speaker label is the same (sequence-only)."""
        try:
            merged: List[TranscriptionSegment] = []

            for start, end, speaker, text in segments:
                text = (text or "").strip()
                if not text:
                    continue

                if merged and merged[-1].speaker == speaker:
                    last = merged[-1]
                    merged[-1] = TranscriptionSegment(
                        start_time=last.start_time,
                        end_time=end,
                        speaker=speaker,
                        text=(last.text + " " + text).strip(),
                    )
                else:
                    merged.append(TranscriptionSegment(start, end, speaker, text))

            self.logger.info(f"Merged {len(segments)} segments into {len(merged)} segments")
            return merged

        except Exception as e:
            self.logger.error(f"Failed to merge segments: {e}")
            raise
    
    def process_audio(self, audio_path: str) -> List[TranscriptionSegment]:
        """Complete audio processing pipeline: diarization + transcription."""
        try:
            # Convert to WAV if needed
            wav_path = self.convert_audio_format(audio_path)
            
            # Perform diarization
            diarization = self.perform_diarization(wav_path)
            
            # Transcribe audio
            transcription_segments = self.transcribe_audio(audio_path)
            
            # Combine diarization with transcription
            combined_segments = []
            for start, end, text in transcription_segments:
                speaker = self._pick_speaker(diarization, start, end)
                combined_segments.append((start, end, speaker, text))
            
            # Merge segments by speaker
            final_segments = self.merge_segments_by_speaker(combined_segments)
            
            self.logger.info(f"Audio processing completed: {len(final_segments)} final segments")
            return final_segments
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self._pipeline:
                del self._pipeline
                self._pipeline = None
            
            if self._whisper_model:
                del self._whisper_model
                self._whisper_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
