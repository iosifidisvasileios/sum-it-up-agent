from __future__ import annotations

import os
import time
import unittest
from typing import Any, Dict, List

import dotenv

from sum_it_up_agent.agent.models import CommunicationChannel, SummaryType
from sum_it_up_agent.agent.prompt_parser import PromptParser

from .test_data import DATASET, SYSTEM_PROMPTS, PromptEvalCase
from .test_utils import (
    _channels_to_values,
    _looks_like_email,
    _summarize_latencies,
    _summary_types_to_values,
    perform_fair_latency_cooldown,
)

dotenv.load_dotenv()


class TestPromptParserEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = "http://localhost:11434"
        cls.fair_latency = os.getenv("PROMPT_EVAL_FAIR_LATENCY", "1").strip().lower() not in {"0", "false", "no"}
        cls.cooldown_ms = int(os.getenv("PROMPT_EVAL_COOLDOWN_MS", "150").strip() or "0")
        cls.models = [
            # "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:BF16",
            # "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest",
            # "hf.co/unsloth/Phi-4-mini-reasoning-GGUF:BF16",
            # "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:BF16",
            # "hf.co/unsloth/gemma-3-27b-it-GGUF:Q2_K_XL",
            # "hf.co/mradermacher/falcon-40b-i1-GGUF:IQ2_S",
            # "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
            # "hf.co/lmstudio-community/EXAONE-3.5-2.4B-Instruct-GGUF:Q8_0",
            # "hf.co/mradermacher/ZYH-LLM-Qwen2.5-14B-V3-GGUF:Q6_K",
            "hf.co/mradermacher/ZYH-LLM-Qwen2.5-14B-V5-GGUF:Q6_K"
        ]

        cls.system_prompt_keys = ['strict_json', 'default', 'conversational', 'step_by_step']

    def test_data_validation(self) -> None:
        """Test that our test data is valid."""
        # Check that all expected channels are valid CommunicationChannel values
        allowed_channels = {c.value for c in CommunicationChannel}
        for case in DATASET:
            if "communication_channels" in case.expected:
                for channel in case.expected["communication_channels"]:
                    self.assertIn(
                        channel,
                        allowed_channels,
                        f"Invalid channel '{channel}' in case '{case.name}'"
                    )

        # Check that all expected summary types are valid SummaryType values
        allowed_types = {t.value for t in SummaryType}
        for case in DATASET:
            if "summary_types_any" in case.expected:
                for summary_type in case.expected["summary_types_any"]:
                    self.assertIn(
                        summary_type,
                        allowed_types,
                        f"Invalid summary type '{summary_type}' in case '{case.name}'"
                    )

    def test_basic_functionality(self) -> None:
        """Test basic parsing functionality with simple cases."""
        simple_cases = [case for case in DATASET if not case.name.startswith("complex_")]

        for case in simple_cases[:3]:  # Test first 3 simple cases
            with self.subTest(case=case.name):
                actual = self._run_case(
                    case=case,
                    model=self.models[0],
                    system_prompt_key='default'
                )

                # Check basic boolean expectations
                if "wants_summary" in case.expected:
                    self.assertEqual(
                        actual["wants_summary"],
                        case.expected["wants_summary"],
                        f"Case {case.name}: wants_summary mismatch"
                    )

                if "wants_transcription" in case.expected:
                    self.assertEqual(
                        actual["wants_transcription"],
                        case.expected["wants_transcription"],
                        f"Case {case.name}: wants_transcription mismatch"
                    )

    def _run_case(
            self,
            *,
            case: PromptEvalCase,
            model: str,
            system_prompt_key: str,
    ) -> Dict[str, Any]:
        system_prompt_text = SYSTEM_PROMPTS.get(system_prompt_key)

        parser = PromptParser(
            provider="ollama",
            model=model,
            base_url=self.base_url,
            prompt_limit=10_000,
            system_prompt_text=system_prompt_text,
        )

        start = time.perf_counter()
        intent = __import__("asyncio").run(parser.parse_prompt(case.prompt))
        latency_ms = (time.perf_counter() - start) * 1000

        if self.fair_latency:
            # Best-effort: encourage Ollama to actually unload/evict the model between measurements.
            # The provider already uses keep_alive=0, but this helps make VRAM release more deterministic.
            perform_fair_latency_cooldown(self.base_url, model, self.cooldown_ms)

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

    def test_eval_matrix(self) -> None:
        failures: List[str] = []

        stats: Dict[str, Dict[str, Any]] = {}
        stats_by_model: Dict[str, Dict[str, Any]] = {}
        stats_by_system_prompt: Dict[str, Dict[str, Any]] = {}
        stats_by_case: Dict[str, Dict[str, Any]] = {}

        for model in self.models:
            for sp_key in self.system_prompt_keys:
                for case in DATASET:
                    actual = self._run_case(case=case, model=model, system_prompt_key=sp_key)
                    exp = case.expected

                    key = f"{model}::{sp_key}"
                    if key not in stats:
                        stats[key] = {
                            "model": model,
                            "system_prompt": sp_key,
                            "total": 0,
                            "passed": 0,
                            "failed": 0,
                            "latencies_ms": [],
                            "failed_cases": [],
                        }
                    stats[key]["total"] += 1
                    stats[key]["latencies_ms"].append(actual["latency_ms"])

                    if model not in stats_by_model:
                        stats_by_model[model] = {"model": model, "total": 0, "passed": 0, "failed": 0,
                                                 "latencies_ms": []}
                    stats_by_model[model]["total"] += 1
                    stats_by_model[model]["latencies_ms"].append(actual["latency_ms"])

                    if sp_key not in stats_by_system_prompt:
                        stats_by_system_prompt[sp_key] = {
                            "system_prompt": sp_key,
                            "total": 0,
                            "passed": 0,
                            "failed": 0,
                            "latencies_ms": [],
                        }
                    stats_by_system_prompt[sp_key]["total"] += 1
                    stats_by_system_prompt[sp_key]["latencies_ms"].append(actual["latency_ms"])

                    if case.name not in stats_by_case:
                        stats_by_case[case.name] = {"case": case.name, "total": 0, "passed": 0, "failed": 0,
                                                    "latencies_ms": []}
                    stats_by_case[case.name]["total"] += 1
                    stats_by_case[case.name]["latencies_ms"].append(actual["latency_ms"])

                    case_failures: List[str] = []

                    # Scalar expectations
                    if "wants_summary" in exp and actual["wants_summary"] != exp["wants_summary"]:
                        case_failures.append(
                            f"{model}/{sp_key}/{case.name}: wants_summary={actual['wants_summary']} expected {exp['wants_summary']}"
                        )
                    if "wants_transcription" in exp and actual["wants_transcription"] != exp["wants_transcription"]:
                        case_failures.append(
                            f"{model}/{sp_key}/{case.name}: wants_transcription={actual['wants_transcription']} expected {exp['wants_transcription']}"
                        )

                    # Channels: check that all expected channels are present
                    if "communication_channels" in exp:
                        missing = [c for c in exp["communication_channels"] if
                                   c not in actual["communication_channels"]]
                        if missing:
                            case_failures.append(
                                f"{model}/{sp_key}/{case.name}: missing channels {missing}; got {actual['communication_channels']}"
                            )

                    # Validate channels are members of CommunicationChannel
                    allowed_channels = {c.value for c in CommunicationChannel}
                    invalid_channels = [c for c in actual["communication_channels"] if c not in allowed_channels]
                    if invalid_channels:
                        case_failures.append(
                            f"{model}/{sp_key}/{case.name}: invalid channels {invalid_channels}; allowed={sorted(allowed_channels)}"
                        )

                    # Recipients contains
                    if "recipients_contains" in exp:
                        for r in exp["recipients_contains"]:
                            if r not in actual["recipients"]:
                                case_failures.append(
                                    f"{model}/{sp_key}/{case.name}: recipients missing {r}; got {actual['recipients']}"
                                )

                    # Summary types any
                    if "summary_types_any" in exp:
                        if not any(t in actual["summary_types"] for t in exp["summary_types_any"]):
                            case_failures.append(
                                f"{model}/{sp_key}/{case.name}: expected one of summary_types {exp['summary_types_any']}; got {actual['summary_types']}"
                            )

                    # Validate summary types are members of SummaryType
                    allowed_types = {t.value for t in SummaryType}
                    invalid_types = [t for t in actual["summary_types"] if t not in allowed_types]
                    if invalid_types:
                        case_failures.append(
                            f"{model}/{sp_key}/{case.name}: invalid summary_types {invalid_types}; allowed={sorted(allowed_types)}"
                        )

                    # Recipient sanity checks
                    if exp.get("recipients_are_emails"):
                        non_emails = [r for r in actual["recipients"] if
                                      isinstance(r, str) and r.strip() and not _looks_like_email(r)]
                        if non_emails:
                            case_failures.append(
                                f"{model}/{sp_key}/{case.name}: expected all recipients to be emails; non_emails={non_emails}; got {actual['recipients']}"
                            )

                    if "custom_instructions_contains_any" in exp:
                        haystack = " ".join(actual.get("custom_instructions") or []).lower()
                        if not any(s.lower() in haystack for s in exp["custom_instructions_contains_any"]):
                            case_failures.append(
                                f"{model}/{sp_key}/{case.name}: expected custom_instructions to contain any of {exp['custom_instructions_contains_any']}; got {actual['custom_instructions']}"
                            )

                    if case_failures:
                        stats[key]["failed"] += 1
                        stats[key]["failed_cases"].append(case.name)
                        failures.extend(case_failures)

                        stats_by_model[model]["failed"] += 1
                        stats_by_system_prompt[sp_key]["failed"] += 1
                        stats_by_case[case.name]["failed"] += 1
                    else:
                        stats[key]["passed"] += 1

                        stats_by_model[model]["passed"] += 1
                        stats_by_system_prompt[sp_key]["passed"] += 1
                        stats_by_case[case.name]["passed"] += 1

                    # Emit per-case metrics (useful in CI logs)
                    print(
                        f"MODEL={model} SYSTEM_PROMPT={sp_key} CASE={case.name} latency_ms={actual['latency_ms']:.1f} "
                        f"channels={actual['communication_channels']} summary_types={actual['summary_types']}"
                    )

        rows: List[str] = []
        rows.append("# PromptParser Eval Report")
        rows.append("")
        rows.append("## Summary by model and system prompt")
        rows.append("")
        rows.append(
            "| model | system_prompt | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
        rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for k in sorted(stats.keys()):
            s = stats[k]
            total = int(s["total"])
            passed = int(s["passed"])
            failed = int(s["failed"])
            pass_rate = (passed / total) if total else 0.0
            lat = [float(x) for x in s["latencies_ms"]]
            lat_summary = _summarize_latencies(lat)

            rows.append(
                f"| {s['model']} | {s['system_prompt']} | {total} | {passed} | {failed} | {pass_rate:.1%} | {lat_summary['avg']:.1f} | {lat_summary['p50']:.1f} | {lat_summary['p95']:.1f} |"
            )

        rows.append("")
        rows.append("## Summary by model")
        rows.append("")
        rows.append(
            "| model | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
        rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for model in sorted(stats_by_model.keys()):
            s = stats_by_model[model]
            total = int(s["total"])
            passed = int(s["passed"])
            failed = int(s["failed"])
            pass_rate = (passed / total) if total else 0.0
            lat_summary = _summarize_latencies([float(x) for x in s["latencies_ms"]])
            rows.append(
                f"| {model} | {total} | {passed} | {failed} | {pass_rate:.1%} | {lat_summary['avg']:.1f} | {lat_summary['p50']:.1f} | {lat_summary['p95']:.1f} |"
            )

        rows.append("")
        rows.append("## Summary by system prompt")
        rows.append("")
        rows.append(
            "| system_prompt | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
        rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for sp_key in sorted(stats_by_system_prompt.keys()):
            s = stats_by_system_prompt[sp_key]
            total = int(s["total"])
            passed = int(s["passed"])
            failed = int(s["failed"])
            pass_rate = (passed / total) if total else 0.0
            lat_summary = _summarize_latencies([float(x) for x in s["latencies_ms"]])
            rows.append(
                f"| {sp_key} | {total} | {passed} | {failed} | {pass_rate:.1%} | {lat_summary['avg']:.1f} | {lat_summary['p50']:.1f} | {lat_summary['p95']:.1f} |"
            )

        rows.append("")
        rows.append("## Summary by test case")
        rows.append("")
        rows.append("| case | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
        rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for case_name in sorted(stats_by_case.keys()):
            s = stats_by_case[case_name]
            total = int(s["total"])
            passed = int(s["passed"])
            failed = int(s["failed"])
            pass_rate = (passed / total) if total else 0.0
            lat_summary = _summarize_latencies([float(x) for x in s["latencies_ms"]])
            rows.append(
                f"| {case_name} | {total} | {passed} | {failed} | {pass_rate:.1%} | {lat_summary['avg']:.1f} | {lat_summary['p50']:.1f} | {lat_summary['p95']:.1f} |"
            )

        if failures:
            rows.append("")
            rows.append("## Failures")
            rows.append("")
            for f in failures:
                rows.append(f"- {f}")

        report = "\n".join(rows)
        print("\n" + report + "\n")

        report_path = os.getenv("PROMPT_EVAL_REPORT_PATH", "").strip()
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
