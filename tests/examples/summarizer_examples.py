#!/usr/bin/env python3
"""
Example usage of the Summarizer library.

This script demonstrates how to use the production-grade meeting summarization
system with different LLM providers and configurations.
"""

import logging
import requests

from sum_it_up_agent.summarizer import (
    SummarizationUseCase,
    SummarizerType,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_ollama_local():
    """Example using Ollama local models."""
    print("\n=== Ollama Local Model Example ===")
    
    # Create use case with Ollama preset
    use_case = SummarizationUseCase.create_with_preset(
        summarizer_type=SummarizerType.OLLAMA_LOCAL
    )
    
    json_path = "/home/vios/PycharmProjects/sum-it-up-agent/examples/ollama_summaries/Product_Marketing_transcription.json"
    
    try:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama is not running or not accessible at http://localhost:11434")
                print("Start Ollama with: ollama serve")
                return

            models = response.json().get("models", [])
            if not models:
                print("❌ No models found in Ollama")
                print("Pull a model with: ollama pull llama2")
                return

            print(f"✅ Ollama is running with {len(models)} models:")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']}")

        except requests.exceptions.RequestException:
            print("❌ Cannot connect to Ollama at http://localhost:11434")
            print("Make sure Ollama is installed and running: ollama serve")
            return

        # Summarize with Ollama
        result = use_case.summarize_transcription_file(

            file_path=json_path,
            meeting_type="planning / coordination meeting",
            output_dir="./ollama_summaries"
        )

        if result.is_successful():
            print(f"✅ Ollama summarization successful!")
            print(f"Model: {result.model_name}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Summary saved to ./ollama_summaries/")

            # Show executive summary
            if result.summary_data and "executive_summary" in result.summary_data:
                exec_summary = result.summary_data["executive_summary"]
                # Truncate long summaries
                if len(exec_summary) > 200:
                    exec_summary = exec_summary[:200] + "..."
                print(f"\nExecutive Summary:\n{exec_summary}")
        else:
            print(f"❌ Ollama summarization failed: {result.error_message}")
    
    except Exception as e:
        print(f"Error with Ollama: {e}")
        print("\nTroubleshooting tips:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        print("4. Check if Ollama is running: curl http://localhost:11434/api/tags")

def main():
    """Main function to run examples."""
    setup_logging()
    
    print("Meeting Summarizer Library Examples")
    print("==================================")

    # Run examples
    try:
        example_ollama_local()

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nCheck the following directories for results:")
    print("- ./ollama_summaries/")


if __name__ == "__main__":
    main()
