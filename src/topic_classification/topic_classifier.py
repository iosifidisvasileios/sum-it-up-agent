import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import pipeline

from .interfaces import (
    ITopicClassifier,
    TopicClassificationConfig,
    ClassificationResult,
    ConversationFormat,
    EnsembleMethod,
    DeviceType
)


class TopicClassifier(ITopicClassifier):
    """Production-grade zero-shot topic classifier for conversations."""
    
    def __init__(self, config: TopicClassificationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pipelines = {}
        self._device = torch.device(config.device.value)
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize zero-shot classification models."""
        try:
            self.logger.info(f"Initializing {len(self.config.models)} models on {self.config.device.value}")
            
            for model_name in self.config.models:
                self.logger.info(f"Loading model: {model_name}")
                classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=self.config.device.value
                )
                self._pipelines[model_name] = classifier
            
            self.logger.info("All models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load and parse JSON data from file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"Loading JSON data from: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.logger.info(f"JSON data loaded successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON data: {e}")
            raise
    
    def extract_segments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation segments from JSON data."""
        try:
            def find_segments_recursive(obj: Any) -> List[Dict[str, Any]]:
                if isinstance(obj, dict):
                    if "segments" in obj and isinstance(obj["segments"], list):
                        return obj["segments"]
                    for value in obj.values():
                        segments = find_segments_recursive(value)
                        if segments:
                            return segments
                elif isinstance(obj, list):
                    for item in obj:
                        segments = find_segments_recursive(item)
                        if segments:
                            return segments
                return []
            
            segments = find_segments_recursive(data)
            
            if not segments:
                raise ValueError("Could not find a 'segments' list in the provided JSON")
            
            self.logger.info(f"Extracted {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to extract segments: {e}")
            raise
    
    def determine_conversation_format(self, segments: List[Dict[str, Any]]) -> ConversationFormat:
        """Determine if conversation is one-on-one or group meeting."""
        try:
            speakers = {segment.get("speaker") for segment in segments if segment.get("speaker")}
            speakers.discard(None)
            
            if len(speakers) <= 2:
                return ConversationFormat.ONE_ON_ONE
            elif len(speakers) > 2:
                return ConversationFormat.GROUP_MEETING
            else:
                return ConversationFormat.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Failed to determine conversation format: {e}")
            return ConversationFormat.UNKNOWN
    
    def build_conversation_text(self, segments: List[Dict[str, Any]]) -> str:
        """Build formatted text from conversation segments."""
        try:
            lines = []
            for segment in segments:
                speaker = segment.get("speaker", "UNKNOWN")
                text = (segment.get("text") or "").strip()
                if text:
                    lines.append(f"{speaker}: {text}")
            
            conversation_text = "\n".join(lines)
            
            if len(conversation_text) < self.config.min_text_length:
                self.logger.warning(f"Conversation text is too short: {len(conversation_text)} characters")
            
            return conversation_text
            
        except Exception as e:
            self.logger.error(f"Failed to build conversation text: {e}")
            raise
    
    def classify_text(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Classify text using zero-shot classification."""
        try:
            if not text.strip():
                raise ValueError("Empty text provided for classification")
            
            results = {}
            
            for model_name, classifier in self._pipelines.items():
                self.logger.debug(f"Classifying with model: {model_name}")
                
                output = classifier(
                    text,
                    candidate_labels=labels,
                    multi_label=self.config.multi_label,
                    hypothesis_template=self.config.hypothesis_template,
                    truncation=self.config.truncation,
                )
                
                # Convert to dictionary with all labels
                model_scores = {label: 0.0 for label in labels}
                for label, score in zip(output["labels"], output["scores"]):
                    model_scores[label] = float(score)
                
                results[model_name] = model_scores
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to classify text: {e}")
            raise
    
    def ensemble_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions from multiple models."""
        try:
            if not predictions:
                raise ValueError("No predictions provided for ensembling")
            
            labels = list(predictions[0].keys())
            ensemble_scores = {label: 0.0 for label in labels}
            
            if self.config.ensemble_method == EnsembleMethod.MEAN:
                # Simple averaging
                for prediction in predictions:
                    for label in labels:
                        ensemble_scores[label] += prediction.get(label, 0.0)
                
                for label in labels:
                    ensemble_scores[label] /= len(predictions)
            
            elif self.config.ensemble_method == EnsembleMethod.MAX:
                # Take maximum score for each label
                for prediction in predictions:
                    for label in labels:
                        ensemble_scores[label] = max(
                            ensemble_scores[label],
                            prediction.get(label, 0.0)
                        )
            
            elif self.config.ensemble_method == EnsembleMethod.WEIGHTED_MEAN:
                # Equal weights for now, could be extended
                weights = [1.0 / len(predictions)] * len(predictions)
                for prediction, weight in zip(predictions, weights):
                    for label in labels:
                        ensemble_scores[label] += prediction.get(label, 0.0) * weight
            
            else:
                # Default to mean
                return self.ensemble_predictions(predictions)
            
            return ensemble_scores
            
        except Exception as e:
            self.logger.error(f"Failed to ensemble predictions: {e}")
            raise
    
    def classify_conversation(self, file_path: str) -> ClassificationResult:
        """Complete classification pipeline for a conversation file."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Classifying conversation: {file_path}")
            
            # Load and extract data
            data = self.load_json_data(file_path)
            all_segments = self.extract_segments(data)
            
            # Limit segments if specified
            if self.config.first_n_segments and self.config.first_n_segments > 0:
                segments = all_segments[:self.config.first_n_segments]
            else:
                segments = all_segments
            
            # Build conversation
            conversation_format = self.determine_conversation_format(segments)
            conversation_text = self.build_conversation_text(segments)
            
            # Classify
            per_model_predictions = self.classify_text(conversation_text, self.config.labels)
            ensemble_scores = self.ensemble_predictions(list(per_model_predictions.values()))
            
            # Get best prediction
            best_topic = max(ensemble_scores, key=ensemble_scores.get)
            confidence = ensemble_scores[best_topic]
            
            # Sort ensemble scores
            sorted_ensemble_scores = dict(
                sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
            )
            
            processing_time = time.time() - start_time
            
            result = ClassificationResult(
                file_path=str(Path(file_path)),
                conversation_format=conversation_format,
                predicted_topic=best_topic,
                confidence=confidence,
                ensemble_scores=sorted_ensemble_scores,
                per_model_scores=per_model_predictions,
                num_segments_used=len(segments),
                num_segments_total=len(all_segments),
                text_length=len(conversation_text),
                processing_time=processing_time
            )
            
            self.logger.info(f"Classification completed: {best_topic} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to classify conversation: {e}")
            raise
    
    def batch_classify(self, file_paths: List[str]) -> List[ClassificationResult]:
        """Classify multiple conversation files."""
        results = []
        
        for file_path in file_paths:
            try:
                result = self.classify_conversation(file_path)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to classify {file_path}: {e}")
                # Continue with other files
                continue
        
        self.logger.info(f"Batch classification completed: {len(results)}/{len(file_paths)} files processed")
        return results
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            for model_name, pipeline in self._pipelines.items():
                del pipeline
                self.logger.debug(f"Cleaned up model: {model_name}")
            
            self._pipelines.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
