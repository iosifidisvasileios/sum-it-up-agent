#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of the MLflow prompt evaluator.

This script shows how to use the MlflowPromptEvaluator class in your own code
for custom experiments and analysis.
"""

import os
import pandas as pd
from mlflow_prompt_eval import MlflowPromptEvaluator, ExperimentConfig


def example_basic_evaluation():
    """Example: Run a basic evaluation with default settings."""
    print("=== Basic Evaluation Example ===")
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name="example_evaluation",
        models=["hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL"],
        system_prompt_keys=["default", "strict_json"],
        fair_latency=True,
        cooldown_ms=100,
    )
    
    # Create evaluator and run experiment
    evaluator = MlflowPromptEvaluator(config)
    results = evaluator.run_experiment()
    
    print(f"Completed evaluation with {len(results)} total results")
    print(f"Results shape: {results.shape}")
    
    return results




def example_analysis_workflow():
    """Example: Complete analysis workflow with custom filtering and insights."""
    print("\n=== Analysis Workflow Example ===")
    
    # Step 1: Run evaluation with multiple configurations
    config = ExperimentConfig(
        experiment_name="analysis_workflow",
        models=[
            "hf.co/lmstudio-community/EXAONE-3.5-2.4B-Instruct-GGUF:Q8_0",
            "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
            # Add more models as needed
        ],
        system_prompt_keys=["default", "strict_json", "conversational"],
        run_tags={"workflow": "analysis_example"}
    )
    
    evaluator = MlflowPromptEvaluator(config)
    results = evaluator.run_experiment()
    
    # Step 2: Analyze results
    print("\nDetailed Analysis:")
    
    # Performance by system prompt
    prompt_performance = results.groupby('system_prompt').agg({
        'passed': ['count', 'sum', 'mean'],
        'latency_ms': ['mean', 'std']
    }).round(3)
    
    prompt_performance.columns = ['total_cases', 'passed_cases', 'pass_rate', 'avg_latency', 'std_latency']
    print("\nPerformance by System Prompt:")
    print(prompt_performance.to_string())
    
    # Find best performing configuration
    best_config = results.loc[results.groupby(['model', 'system_prompt'])['passed'].transform('sum').idxmax()]
    print(f"\nBest performing configuration:")
    print(f"  Model: {best_config['model']}")
    print(f"  System Prompt: {best_config['system_prompt']}")
    print(f"  Pass Rate: {results[(results['model'] == best_config['model']) & (results['system_prompt'] == best_config['system_prompt'])]['passed'].mean():.1%}")
    
    # Analyze failure patterns
    failures = results[~results['passed']]
    if not failures.empty:
        print(f"\nFailure Analysis ({len(failures)} failed cases):")
        
        # Most common failure cases
        common_failures = failures['case_name'].value_counts().head(5)
        print("Most common failing test cases:")
        for case_name, count in common_failures.items():
            print(f"  {case_name}: {count} failures")
        
        # Failure reasons by case
        for case_name in common_failures.index[:3]:  # Top 3
            case_failures = failures[failures['case_name'] == case_name]
            print(f"\nFailure details for {case_name}:")
            for _, row in case_failures.iterrows():
                print(f"  {row['system_prompt']}: {row['failures']}")
    
    return results, prompt_performance


def example_experiment_comparison():
    """Example: Compare results across different experiments."""
    print("\n=== Cross-Experiment Comparison Example ===")
    
    # Run multiple experiments with different configurations
    experiments = [
        {
            "name": "fast_evaluation",
            "models": ["hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL"],
            "prompts": ["minimal_1", "minimal_2"],
            "fair_latency": False,
            "cooldown_ms": 50
        },
        {
            "name": "thorough_evaluation", 
            "models": ["hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL"],
            "prompts": ["default", "strict_json", "conversational"],
            "fair_latency": True,
            "cooldown_ms": 200
        }
    ]
    
    all_results = []
    
    for exp_config in experiments:
        print(f"\nRunning experiment: {exp_config['name']}")
        
        config = ExperimentConfig(
            experiment_name=exp_config["name"],
            models=exp_config["models"],
            system_prompt_keys=exp_config["prompts"],
            fair_latency=exp_config["fair_latency"],
            cooldown_ms=exp_config["cooldown_ms"],
            run_tags={"experiment_type": exp_config["name"]}
        )
        
        evaluator = MlflowPromptEvaluator(config)
        results = evaluator.run_experiment()
        results['experiment'] = exp_config['name']
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Compare experiments
    exp_comparison = combined_results.groupby(['experiment', 'system_prompt']).agg({
        'passed': ['count', 'sum', 'mean'],
        'latency_ms': ['mean', 'median']
    }).round(3)
    
    exp_comparison.columns = ['total_cases', 'passed_cases', 'pass_rate', 'avg_latency', 'median_latency']
    print("\nCross-Experiment Comparison:")
    print(exp_comparison.to_string())
    
    return combined_results, exp_comparison


def save_results_to_files(results: pd.DataFrame, comparison: pd.DataFrame, prefix: str = "example"):
    """Save results to local files for further analysis."""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_path = f"/tmp/{prefix}_results_{timestamp}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save comparison
    comparison_path = f"/tmp/{prefix}_comparison_{timestamp}.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Comparison saved to: {comparison_path}")
    
    return results_path, comparison_path


def main():
    """Run all examples."""
    print("MLflow Prompt Evaluator - Example Usage")
    print("=" * 50)
    
    try:
        # Example 1: Basic evaluation
        # basic_results = example_basic_evaluation()
        
        # Example 3: Analysis workflow
        analysis_results, performance_summary = example_analysis_workflow()
        
        # Example 4: Cross-experiment comparison
        # combined_results, exp_comparison = example_experiment_comparison()
        
        # Save results
        save_results_to_files(analysis_results, performance_summary, "analysis")
        # save_results_to_files(combined_results, exp_comparison, "cross_experiment")
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check MLflow UI at http://localhost:5000 for detailed tracking")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure Ollama is running and models are available")
        raise


if __name__ == "__main__":
    main()
