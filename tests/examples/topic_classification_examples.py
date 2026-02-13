#!/usr/bin/env python3
"""
Example usage of the Topic Classification library.

This script demonstrates how to use the production-grade topic classification
system with different configurations and use cases.
"""

import logging

from sum_it_up_agent.topic_classification import (
    TopicClassificationUseCase,
    TopicClassificationConfig,
    DeviceType,
    EnsembleMethod
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )



def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration matching your notebook exactly
    custom_config = TopicClassificationConfig(
        device=DeviceType.CUDA,
        models=[
            "FacebookAI/roberta-large-mnli",
            "cross-encoder/nli-deberta-v3-small",
            "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        ],
        labels=[
            "team status sync / standup",
            "planning / coordination meeting",
            "decision-making meeting",
            "brainstorming session",
            "retrospective / postmortem",
            "training / onboarding",
            "interview",
            "customer call / sales demo",
            "support / incident call",
            "other",
        ],
        hypothesis_template="This conversation is about {}.",
        multi_label=True,
        truncation=True,
        ensemble_method=EnsembleMethod.MEAN,
        first_n_segments=50,  # Same as your notebook
    )
    
    # Create use case with custom config
    use_case = TopicClassificationUseCase.create_with_custom_classifier(custom_config)
    output_dir = "./topic_classification_results.json"
    json_path = "/home/vios/PycharmProjects/sum-it-up-agent/examples/output/Product_Marketing_transcription.json"
    
    try:
        result = use_case.classify_single_file(json_path)

        print(f"Custom configuration results:")
        print(f"Topic: {result.predicted_topic}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Models used: {len(result.per_model_scores)}")

        # Show per-model scores
        print(f"\nPer-model scores:")
        for model, scores in result.per_model_scores.items():
            top_score = max(scores.items(), key=lambda x: x[1])
            print(f"  {model}: {top_score[0]} ({top_score[1]:.4f})")

        use_case.export_results([result], output_dir, format_type="json", include_analysis=False)


    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run examples."""
    setup_logging()
    
    print("Topic Classification Library Examples")
    print("====================================")
    
    # Run examples
    try:
        example_custom_configuration()

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
