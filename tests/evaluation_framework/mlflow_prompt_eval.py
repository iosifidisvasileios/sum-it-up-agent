#!/usr/bin/env python3
"""
MLflow-based prompt and LLM evaluation script.

This script uses MLflow to track and compare different prompts and LLM models
for the PromptParser task. It runs the same test dataset across different
combinations of models and system prompts, logging metrics and artifacts to MLflow.
"""

from __future__ import annotations

import time
import argparse
from typing import Any, Dict, List
from dataclasses import dataclass

import dotenv
import mlflow
import mlflow.pytorch
import pandas as pd
from mlflow.tracking import MlflowClient

from sum_it_up_agent.agent.models import CommunicationChannel, SummaryType
from sum_it_up_agent.agent.prompt_parser import PromptParser

from tests.unittests.test_data import DATASET, SYSTEM_PROMPTS, PromptEvalCase
from tests.unittests.test_utils import (
    _channels_to_values,
    _looks_like_email,
    _summarize_latencies,
    _summary_types_to_values,
    perform_fair_latency_cooldown,
)

dotenv.load_dotenv()


@dataclass
class ExperimentConfig:
    """Configuration for the MLflow experiment."""
    experiment_name: str = "prompt_parser_evaluation"
    base_url: str = "http://localhost:11434"
    models: List[str] = None
    system_prompt_keys: List[str] = None
    fair_latency: bool = True
    cooldown_ms: int = 150
    run_tags: Dict[str, str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = [
                "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
                # Add more models as needed
            ]
        
        if self.system_prompt_keys is None:
            self.system_prompt_keys = [
                'strict_json', 'default', 'conversational', 'step_by_step', 
                'minimal_1', 'minimal_2', 'minimal_3'
            ]
        
        if self.run_tags is None:
            self.run_tags = {}


class MlflowPromptEvaluator:
    """MLflow-based prompt and model evaluator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MlflowClient()
        
        # Set up MLflow experiment
        mlflow.set_experiment(config.experiment_name)
        
        # Validate test data
        self._validate_test_data()
    
    def _validate_test_data(self) -> None:
        """Validate that test data is consistent with model expectations."""
        # Check that all expected channels are valid CommunicationChannel values
        allowed_channels = {c.value for c in CommunicationChannel}
        for case in DATASET:
            if "communication_channels" in case.expected:
                for channel in case.expected["communication_channels"]:
                    if channel not in allowed_channels:
                        raise ValueError(f"Invalid channel '{channel}' in case '{case.name}'")
        
        # Check that all expected summary types are valid SummaryType values
        allowed_types = {t.value for t in SummaryType}
        for case in DATASET:
            if "summary_types_any" in case.expected:
                for summary_type in case.expected["summary_types_any"]:
                    if summary_type not in allowed_types:
                        raise ValueError(f"Invalid summary type '{summary_type}' in case '{case.name}'")
    
    def _run_single_case(
        self,
        case: PromptEvalCase,
        model: str,
        system_prompt_key: str,
    ) -> Dict[str, Any]:
        """Run a single test case and return results."""
        system_prompt_text = SYSTEM_PROMPTS.get(system_prompt_key)
        
        parser = PromptParser(
            provider="ollama",
            model=model,
            base_url=self.config.base_url,
            prompt_limit=10_000,
            system_prompt_text=system_prompt_text,
        )
        
        start = time.perf_counter()
        intent = __import__("asyncio").run(parser.parse_prompt(case.prompt))
        latency_ms = (time.perf_counter() - start) * 1000
        
        if self.config.fair_latency:
            perform_fair_latency_cooldown(self.config.base_url, model, self.config.cooldown_ms)
        
        actual = {
            "wants_summary": intent.wants_summary,
            "wants_transcription": intent.wants_transcription,
            "communication_channels": _channels_to_values(intent.communication_channels),
            "recipients": intent.recipients,
            "subject": intent.subject,
            "summary_types": _summary_types_to_values(intent.summary_types),
            "custom_instructions": intent.custom_instructions,
            "latency_ms": latency_ms,
        }
        
        return actual
    
    def _evaluate_case(
        self,
        case: PromptEvalCase,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single case and return metrics."""
        metrics = {
            "passed": True,
            "failures": [],
        }
        
        # Scalar expectations
        if "wants_summary" in expected:
            if actual["wants_summary"] != expected["wants_summary"]:
                metrics["passed"] = False
                metrics["failures"].append(
                    f"wants_summary={actual['wants_summary']} expected {expected['wants_summary']}"
                )
        
        if "wants_transcription" in expected:
            if actual["wants_transcription"] != expected["wants_transcription"]:
                metrics["passed"] = False
                metrics["failures"].append(
                    f"wants_transcription={actual['wants_transcription']} expected {expected['wants_transcription']}"
                )
        
        # Channels: check that all expected channels are present
        if "communication_channels" in expected:
            missing = [c for c in expected["communication_channels"] 
                      if c not in actual["communication_channels"]]
            if missing:
                metrics["passed"] = False
                metrics["failures"].append(
                    f"missing channels {missing}; got {actual['communication_channels']}"
                )
        
        # Validate channels are members of CommunicationChannel
        allowed_channels = {c.value for c in CommunicationChannel}
        invalid_channels = [c for c in actual["communication_channels"] 
                           if c not in allowed_channels]
        if invalid_channels:
            metrics["passed"] = False
            metrics["failures"].append(
                f"invalid channels {invalid_channels}; allowed={sorted(allowed_channels)}"
            )
        
        # Recipients contains
        if "recipients_contains" in expected:
            for r in expected["recipients_contains"]:
                if r not in actual["recipients"]:
                    metrics["passed"] = False
                    metrics["failures"].append(
                        f"recipients missing {r}; got {actual['recipients']}"
                    )
        
        # Summary types any
        if "summary_types_any" in expected:
            if not any(t in actual["summary_types"] for t in expected["summary_types_any"]):
                metrics["passed"] = False
                metrics["failures"].append(
                    f"expected one of summary_types {expected['summary_types_any']}; got {actual['summary_types']}"
                )
        
        # Validate summary types are members of SummaryType
        allowed_types = {t.value for t in SummaryType}
        invalid_types = [t for t in actual["summary_types"] if t not in allowed_types]
        if invalid_types:
            metrics["passed"] = False
            metrics["failures"].append(
                f"invalid summary_types {invalid_types}; allowed={sorted(allowed_types)}"
            )
        
        # Recipient sanity checks
        if expected.get("recipients_are_emails"):
            non_emails = [r for r in actual["recipients"] 
                         if isinstance(r, str) and r.strip() and not _looks_like_email(r)]
            if non_emails:
                metrics["passed"] = False
                metrics["failures"].append(
                    f"expected all recipients to be emails; non_emails={non_emails}; got {actual['recipients']}"
                )
        
        # Custom instructions contains any
        if "custom_instructions_contains_any" in expected:
            haystack = " ".join(actual.get("custom_instructions") or []).lower()
            if not any(s.lower() in haystack for s in expected["custom_instructions_contains_any"]):
                metrics["passed"] = False
                metrics["failures"].append(
                    f"expected custom_instructions to contain any of {expected['custom_instructions_contains_any']}; got {actual['custom_instructions']}"
                )
        
        return metrics
    
    def run_experiment(self) -> pd.DataFrame:
        """Run the complete experiment and return results as a DataFrame."""
        all_results = []
        
        for model in self.config.models:
            for sp_key in self.config.system_prompt_keys:
                print(f"\n=== Running: {model} + {sp_key} ===")
                
                with mlflow.start_run(
                    run_name=f"{model.split('/')[-1]}_{sp_key}",
                    tags={
                        "model": model,
                        "system_prompt": sp_key,
                        **self.config.run_tags
                    }
                ) as run:
                    # Log parameters
                    mlflow.log_params({
                        "model": model,
                        "system_prompt": sp_key,
                        "base_url": self.config.base_url,
                        "fair_latency": self.config.fair_latency,
                        "cooldown_ms": self.config.cooldown_ms,
                        "total_cases": len(DATASET),
                    })
                    
                    # Log system prompt as artifact
                    system_prompt_text = SYSTEM_PROMPTS.get(sp_key, "")
                    with open(f"/tmp/system_prompt_{sp_key}.txt", "w") as f:
                        f.write(system_prompt_text)
                    mlflow.log_artifact(f"/tmp/system_prompt_{sp_key}.txt", "system_prompts")
                    
                    # Run all test cases
                    case_results = []
                    passed_count = 0
                    failed_count = 0
                    latencies = []
                    all_failures = []
                    
                    for case in DATASET:
                        print(f"  Running case: {case.name}")
                        
                        # Run the case
                        actual = self._run_single_case(case, model, sp_key)
                        
                        # Evaluate results
                        evaluation = self._evaluate_case(case, actual, case.expected)
                        
                        # Store results
                        case_result = {
                            "case_name": case.name,
                            "model": model,
                            "system_prompt": sp_key,
                            "passed": evaluation["passed"],
                            "latency_ms": actual["latency_ms"],
                            "wants_summary": actual["wants_summary"],
                            "wants_transcription": actual["wants_transcription"],
                            "communication_channels": actual["communication_channels"],
                            "recipients": actual["recipients"],
                            "summary_types": actual["summary_types"],
                            "custom_instructions": actual["custom_instructions"],
                            "failures": evaluation["failures"],
                        }
                        case_results.append(case_result)
                        
                        # Update counters
                        if evaluation["passed"]:
                            passed_count += 1
                        else:
                            failed_count += 1
                            all_failures.extend([f"{case.name}: {f}" for f in evaluation["failures"]])
                        
                        latencies.append(actual["latency_ms"])
                        
                        # Log per-case metrics
                        mlflow.log_metric(f"case_{case.name}_passed", int(evaluation["passed"]))
                        mlflow.log_metric(f"case_{case.name}_latency_ms", actual["latency_ms"])
                    
                    # Calculate summary metrics
                    total_cases = len(DATASET)
                    pass_rate = passed_count / total_cases if total_cases > 0 else 0.0
                    latency_summary = _summarize_latencies(latencies)
                    
                    # Log summary metrics
                    mlflow.log_metrics({
                        "total_cases": total_cases,
                        "passed_count": passed_count,
                        "failed_count": failed_count,
                        "pass_rate": pass_rate,
                        "latency_avg_ms": latency_summary["avg"],
                        "latency_p50_ms": latency_summary["p50"],
                        "latency_p95_ms": latency_summary["p95"],
                    })
                    
                    # Log detailed results as artifact
                    results_df = pd.DataFrame(case_results)
                    results_path = f"/tmp/results_{model.split('/')[-1]}_{sp_key}.csv"
                    results_df.to_csv(results_path, index=False)
                    mlflow.log_artifact(results_path, "results")
                    
                    # Log failures as artifact if any
                    if all_failures:
                        failures_path = f"/tmp/failures_{model.split('/')[-1]}_{sp_key}.txt"
                        with open(failures_path, "w") as f:
                            f.write("\n".join(all_failures))
                        mlflow.log_artifact(failures_path, "failures")
                    
                    print(f"  Pass rate: {pass_rate:.1%} ({passed_count}/{total_cases})")
                    print(f"  Latency: avg={latency_summary['avg']:.1f}ms, p50={latency_summary['p50']:.1f}ms, p95={latency_summary['p95']:.1f}ms")
                    
                    all_results.extend(case_results)
        
        # Create overall results DataFrame
        overall_df = pd.DataFrame(all_results)
        
        # Log overall summary
        self._log_overall_summary(overall_df)
        
        return overall_df
    
    def _log_overall_summary(self, results_df: pd.DataFrame) -> None:
        """Log an overall summary of the experiment."""
        with mlflow.start_run(run_name="overall_summary") as run:
            # Group by model and system prompt
            summary = results_df.groupby(['model', 'system_prompt']).agg({
                'passed': ['count', 'sum'],
                'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)]
            }).round(2)
            
            summary.columns = ['total_cases', 'passed_cases', 'latency_avg', 'latency_median', 'latency_p95']
            summary['pass_rate'] = (summary['passed_cases'] / summary['total_cases']).round(3)
            
            # Log summary as artifact
            summary_path = "/tmp/overall_summary.csv"
            summary.to_csv(summary_path)
            mlflow.log_artifact(summary_path, "summary")
            
            print(f"\n=== Overall Summary ===")
            print(summary.to_string())
    
    def compare_runs(self, run_ids: List[str] = None) -> pd.DataFrame:
        """Compare specific runs or all runs in the experiment."""
        if run_ids is None:
            # Get all runs in the experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                # No experiment exists, return empty DataFrame
                return pd.DataFrame(columns=[
                    'run_id', 'model', 'system_prompt', 'pass_rate', 'latency_avg_ms',
                    'latency_p50_ms', 'latency_p95_ms', 'total_cases', 'passed_count', 'failed_count'
                ])
            
            runs = mlflow.search_runs(experiment.experiment_id)
            if runs.empty:
                # No runs in experiment, return empty DataFrame
                return pd.DataFrame(columns=[
                    'run_id', 'model', 'system_prompt', 'pass_rate', 'latency_avg_ms',
                    'latency_p50_ms', 'latency_p95_ms', 'total_cases', 'passed_count', 'failed_count'
                ])
            
            run_ids = runs['run_id'].tolist()
        
        comparison_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            data = {
                'run_id': run_id,
                'model': run.data.params.get('model', 'unknown'),
                'system_prompt': run.data.params.get('system_prompt', 'unknown'),
                'pass_rate': run.data.metrics.get('pass_rate', 0),
                'latency_avg_ms': run.data.metrics.get('latency_avg_ms', 0),
                'latency_p50_ms': run.data.metrics.get('latency_p50_ms', 0),
                'latency_p95_ms': run.data.metrics.get('latency_p95_ms', 0),
                'total_cases': run.data.metrics.get('total_cases', 0),
                'passed_count': run.data.metrics.get('passed_count', 0),
                'failed_count': run.data.metrics.get('failed_count', 0),
            }
            comparison_data.append(data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Only sort if we have data
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(['pass_rate', 'latency_avg_ms'], ascending=[False, True])
        
        return comparison_df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run MLflow-based prompt and LLM evaluation")
    parser.add_argument(
        "--experiment-name",
        default="prompt_parser_evaluation",
        help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL"],
        help="List of models to evaluate"
    )
    parser.add_argument(
        "--system-prompts",
        nargs="+",
        default=['default', 'strict_json', 'conversational', 'step_by_step', 'minimal_1', 'minimal_2', 'minimal_3'],
        help="List of system prompt keys to evaluate"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama API"
    )
    parser.add_argument(
        "--no-fair-latency",
        action="store_true",
        help="Disable fair latency cooldown"
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=150,
        help="Cooldown time in milliseconds between runs"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing runs, don't run new experiments"
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        base_url=args.base_url,
        models=args.models,
        system_prompt_keys=args.system_prompts,
        fair_latency=not args.no_fair_latency,
        cooldown_ms=args.cooldown_ms,
    )
    
    # Create evaluator
    evaluator = MlflowPromptEvaluator(config)
    
    if args.compare_only:
        # Only compare existing runs
        comparison_df = evaluator.compare_runs()
        print("\n=== Run Comparison ===")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = "/tmp/run_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
    else:
        # Run the experiment
        print(f"Starting experiment: {config.experiment_name}")
        print(f"Models: {config.models}")
        print(f"System prompts: {config.system_prompt_keys}")
        print(f"Base URL: {config.base_url}")
        
        results_df = evaluator.run_experiment()
        
        # Show comparison
        comparison_df = evaluator.compare_runs()
        print("\n=== Final Comparison ===")
        print(comparison_df.to_string(index=False))
        
        # Save results
        results_path = "/tmp/full_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nFull results saved to: {results_path}")
        
        comparison_path = "/tmp/final_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
