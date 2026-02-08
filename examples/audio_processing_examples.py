#!/usr/bin/env python3
"""
Example usage of the Audio Processor library.

This script demonstrates how to use the production-grade audio processing
system with different configurations and use cases.
"""

import os
import logging
import dotenv

dotenv.load_dotenv()
from sum_it_up_agent.audio_processor import (
    AudioProcessingUseCase,
    ProcessorType,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """Basic usage example with preset configuration."""
    print("=== Basic Usage Example ===")
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        return
    
    # Create use case with standard preset
    use_case = AudioProcessingUseCase.create_with_preset(
        processor_type=ProcessorType.STANDARD,
        huggingface_token=hf_token
    )
    
    # Process audio file
    audio_path = "/home/vios/Downloads/Product_Marketing.mp3"
    
    try:
        segments = use_case.process_audio_file(
            audio_path=audio_path,
            output_format="json",
            save_to_file=True,
            output_dir="./output"
        )

        # Print summary
        summary = use_case.get_transcription_summary(segments)
        print(f"Processing completed:")
        print(f"- Total segments: {summary['total_segments']}")
        print(f"- Total duration: {summary['total_duration']:.2f}s")
        print(f"- Speakers: {summary['speakers']}")
        print(f"- Word count: {summary['word_count']}")

        # Print first few segments
        print("\nFirst 5 segments:")
        for i, segment in enumerate(segments[:5]):
            print(f"[{segment.start_time:.2f}-{segment.end_time:.2f}] {segment.speaker}: {segment.text}")

    except Exception as e:
        print(f"Error processing audio: {e}")

def main():
    """Main function to run examples."""
    setup_logging()
    
    print("Audio Processor Library Examples")
    print("================================")
    
    # Run examples
    try:
        example_basic_usage()
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\nExamples completed!")

if __name__ == "__main__":
    main()
