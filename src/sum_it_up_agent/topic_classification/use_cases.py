import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv

from sum_it_up_agent.topic_classification import ITopicClassifier, ClassificationResult, ClassifierType
from sum_it_up_agent.topic_classification import TopicClassifierFactory


class TopicClassificationUseCase:
    """Use case class for topic classification business logic."""
    
    def __init__(
        self,
        classifier: Optional[ITopicClassifier] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.classifier = classifier
        self.logger = logger or logging.getLogger(__name__)
    
    def classify_single_file(self, file_path: str) -> ClassificationResult:
        """Classify a single conversation file."""
        try:
            self.logger.info(f"Classifying single file: {file_path}")
            result = self.classifier.classify_conversation(file_path)
            
            self.logger.info(f"Classification completed: {result.predicted_topic}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to classify file {file_path}: {e}")
            raise
    
    def classify_multiple_files(self, file_paths: List[str]) -> List[ClassificationResult]:
        """Classify multiple conversation files."""
        try:
            self.logger.info(f"Classifying {len(file_paths)} files")
            results = self.classifier.batch_classify(file_paths)
            
            successful_count = len(results)
            failed_count = len(file_paths) - successful_count
            
            self.logger.info(f"Batch classification completed: {successful_count} successful, {failed_count} failed")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to classify multiple files: {e}")
            raise
    
    def analyze_classification_results(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """Analyze classification results and generate statistics."""
        if not results:
            return {"total_files": 0, "error": "No results to analyze"}
        
        # Basic statistics
        total_files = len(results)
        total_segments = sum(r.num_segments_total for r in results)
        total_processing_time = sum(r.processing_time or 0 for r in results)
        
        # Topic distribution
        topic_counts = {}
        topic_confidences = {}
        
        for result in results:
            topic = result.predicted_topic
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            if topic not in topic_confidences:
                topic_confidences[topic] = []
            topic_confidences[topic].append(result.confidence)
        
        # Calculate average confidences per topic
        avg_confidences = {}
        for topic, confidences in topic_confidences.items():
            avg_confidences[topic] = sum(confidences) / len(confidences)
        
        # Format distribution
        format_counts = {
            "one_on_one": 0,
            "group_meeting": 0,
            "unknown": 0
        }
        
        for result in results:
            format_counts[result.conversation_format.value] += 1
        
        # Performance metrics
        avg_processing_time = total_processing_time / total_files if total_files > 0 else 0
        avg_segments_per_file = total_segments / total_files if total_files > 0 else 0
        
        return {
            "total_files": total_files,
            "total_segments": total_segments,
            "total_processing_time": total_processing_time,
            "avg_processing_time": avg_processing_time,
            "avg_segments_per_file": avg_segments_per_file,
            "topic_distribution": topic_counts,
            "topic_confidences": avg_confidences,
            "format_distribution": format_counts,
            "most_common_topic": max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None,
            "avg_confidence": sum(r.confidence for r in results) / total_files if total_files > 0 else 0
        }
    
    def export_results(
        self,
        results: List[ClassificationResult],
        output_path: str,
        format_type: str = "json",
        include_analysis: bool = True
    ) -> None:
        """Export classification results to different formats."""
        output_path = Path(output_path)
        
        if format_type.lower() == "json":
            self._export_json(results, output_path, include_analysis)
        elif format_type.lower() == "csv":
            self._export_csv(results, output_path)
        elif format_type.lower() == "txt":
            self._export_txt(results, output_path, include_analysis)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Results exported to: {output_path}")
    
    def _export_json(
        self,
        results: List[ClassificationResult],
        output_path: Path,
        include_analysis: bool
    ) -> None:
        """Export results as JSON format."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_results": len(results),
                "classifier_config": self._get_config_summary()
            }
        }
        
        if include_analysis:
            data["analysis"] = self.analyze_classification_results(results)
        
        data["results"] = [
            {
                "file_path": result.file_path,
                "conversation_format": result.conversation_format.value,
                "predicted_topic": result.predicted_topic,
                "confidence": result.confidence,
                "ensemble_scores": result.ensemble_scores,
                "num_segments_used": result.num_segments_used,
                "num_segments_total": result.num_segments_total,
                "text_length": result.text_length,
                "processing_time": result.processing_time
            }
            for result in results
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, results: List[ClassificationResult], output_path: Path) -> None:
        """Export results as CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'file_path', 'conversation_format', 'predicted_topic', 'confidence',
                'num_segments_used', 'num_segments_total', 'text_length', 'processing_time',
                'top_3_scores'
            ])
            
            # Data rows
            for result in results:
                top_3_scores = ", ".join([
                    f"{k}:{v:.3f}" for k, v in list(result.ensemble_scores.items())[:3]
                ])
                
                writer.writerow([
                    result.file_path,
                    result.conversation_format.value,
                    result.predicted_topic,
                    result.confidence,
                    result.num_segments_used,
                    result.num_segments_total,
                    result.text_length,
                    result.processing_time,
                    top_3_scores
                ])
    
    def _export_txt(
        self,
        results: List[ClassificationResult],
        output_path: Path,
        include_analysis: bool
    ) -> None:
        """Export results as plain text format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Topic Classification Results\n")
            f.write("=" * 50 + "\n\n")
            
            if include_analysis:
                analysis = self.analyze_classification_results(results)
                f.write("Summary Analysis:\n")
                f.write(f"Total files: {analysis['total_files']}\n")
                f.write(f"Average confidence: {analysis['avg_confidence']:.3f}\n")
                f.write(f"Most common topic: {analysis['most_common_topic']}\n")
                f.write(f"Average processing time: {analysis['avg_processing_time']:.2f}s\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 50 + "\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\n{i}. {Path(result.file_path).name}\n")
                f.write(f"   Topic: {result.predicted_topic}\n")
                f.write(f"   Confidence: {result.confidence:.3f}\n")
                f.write(f"   Format: {result.conversation_format.value}\n")
                f.write(f"   Segments: {result.num_segments_used}/{result.num_segments_total}\n")
                f.write(f"   Processing time: {result.processing_time:.2f}s\n")
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of classifier configuration."""
        if hasattr(self.classifier, 'config'):
            config = self.classifier.config
            return {
                'device': config.device.value,
                'models': config.models,
                'ensemble_method': config.ensemble_method.value,
                'first_n_segments': config.first_n_segments,
            }
        return {}
    
    def find_files_by_pattern(
        self,
        directory: str,
        pattern: str = "*.json"
    ) -> List[str]:
        """Find files matching a pattern in directory."""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            files = list(directory_path.glob(pattern))
            file_paths = [str(f) for f in files if f.is_file()]
            
            self.logger.info(f"Found {len(file_paths)} files matching pattern '{pattern}'")
            return file_paths
            
        except Exception as e:
            self.logger.error(f"Failed to find files: {e}")
            raise
    
    def classify_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        recursive: bool = False
    ) -> List[ClassificationResult]:
        """Classify all files in a directory matching pattern."""
        try:
            directory_path = Path(directory)
            
            if recursive:
                files = list(directory_path.rglob(pattern))
            else:
                files = list(directory_path.glob(pattern))
            
            file_paths = [str(f) for f in files if f.is_file()]
            
            if not file_paths:
                self.logger.warning(f"No files found matching pattern '{pattern}' in {directory}")
                return []
            
            return self.classify_multiple_files(file_paths)
            
        except Exception as e:
            self.logger.error(f"Failed to classify directory: {e}")
            raise
    
    @classmethod
    def create_with_preset(
        cls,
        classifier_type: ClassifierType = ClassifierType.STANDARD,
        config_overrides: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> 'TopicClassificationUseCase':
        """Create use case with preset classifier."""
        classifier = TopicClassifierFactory.create_classifier(
            classifier_type=classifier_type,
            config_overrides=config_overrides,
            logger=logger,
        )
        return cls(classifier, logger=logger)
    
    @classmethod
    def create_with_custom_classifier(cls, config, logger: Optional[logging.Logger] = None) -> 'TopicClassificationUseCase':
        """Create use case with custom classifier."""
        classifier = TopicClassifierFactory.create_custom_classifier(config, logger=logger)
        return cls(classifier, logger=logger)
