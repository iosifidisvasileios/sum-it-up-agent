from __future__ import annotations

import gc
import os
import re
import time
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dotenv

from sum_it_up_agent.agent.models import CommunicationChannel, SummaryType
from sum_it_up_agent.agent.prompt_parser import PromptParser

dotenv.load_dotenv()


@dataclass(frozen=True)
class PromptEvalCase:
    name: str
    prompt: str
    expected: Dict[str, Any]


DATASET: List[PromptEvalCase] = [
    PromptEvalCase(
        name="summary_only",
        prompt="Summarize this meeting. No email, no pdf.",
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": [],
        },
    ),
    PromptEvalCase(
        name="summary_pdf",
        prompt="Just summarize this, and create a pdf report.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf"],
        },
    ),
    PromptEvalCase(
        name="transcription_only",
        prompt="Only transcribe it. Do not summarize.",
        expected={
            "wants_summary": False,
            "wants_transcription": True,
        },
    ),
    PromptEvalCase(
        name="email_with_recipient",
        prompt="Summarize it and email it to bill@example.com with subject Weekly sync.",
        expected={
            "wants_summary": True,
            "communication_channels": ["email"],
            "recipients_contains": ["bill@example.com"],
        },
    ),
    PromptEvalCase(
        name="action_items_and_decisions",
        prompt="Give me action items and decisions, then generate a PDF.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf"],
            "summary_types_any": ["action_items", "decisions"],
        },
    ),
    PromptEvalCase(
        name="summary_and_transcription",
        prompt="I want both a transcript and a summary.",
        expected={
            "wants_summary": True,
            "wants_transcription": True,
        },
    ),
    PromptEvalCase(
        name="explicit_no_transcription",
        prompt="Summarize it but do NOT include transcription.",
        expected={
            "wants_summary": True,
            "wants_transcription": False,
        },
    ),
    PromptEvalCase(
        name="email_multiple_recipients",
        prompt="Please summarize and email to alice@example.com and bob@example.com.",
        expected={
            "wants_summary": True,
            "communication_channels": ["email"],
            "recipients_contains": ["alice@example.com", "bob@example.com"],
        },
    ),
    PromptEvalCase(
        name="pdf_and_email",
        prompt="Summarize it, export a PDF, and email it to ops@example.com.",
        expected={
            "wants_summary": True,
            "communication_channels": ["pdf", "email"],
            "recipients_contains": ["ops@example.com"],
        },
    ),
    PromptEvalCase(
        name="executive_summary",
        prompt="Give me an executive summary. No email.",
        expected={
            "wants_summary": True,
            "communication_channels": [],
            "summary_types_any": ["executive"],
        },
    ),
    PromptEvalCase(
        name="bullet_points",
        prompt="Summarize in bullet points and also include key points.",
        expected={
            "wants_summary": True,
            "summary_types_any": ["bullet_points", "key_points"],
        },
    ),
    PromptEvalCase(
        name="custom_instructions_tone",
        prompt="Summarize, but keep it concise and professional. Add a short risks section.",
        expected={
            "wants_summary": True,
            "custom_instructions_contains_any": ["concise", "professional", "risks"],
        },
    ),
    PromptEvalCase(
        name="no_summary_no_transcription",
        prompt="Don't summarize and don't transcribe.",
        expected={
            "wants_summary": False,
            "wants_transcription": False,
        },
    ),
    PromptEvalCase(
        name="complex_customer_followup_multi_channel",
        prompt=(
            "We just finished a 55-minute customer call about the new rollout.\n"
            "Please do the following:\n"
            "1) Create a detailed summary with key points, decisions, and action items.\n"
            "2) Export the summary as a PDF.\n"
            "3) Email the PDF and a short executive recap to customer-success@example.com and pm@example.com.\n"
            "Subject: ACME rollout follow-up - next steps\n"
            "Also: keep the tone professional and concise, and highlight any risks/blockers."
        ),
        expected={
            "wants_summary": True,
            "communication_channels": ["email", "pdf"],
            "recipients_contains": ["customer-success@example.com", "pm@example.com"],
            "recipients_are_emails": True,
            "summary_types_any": ["detailed", "key_points", "decisions", "action_items", "executive"],
            "custom_instructions_contains_any": ["professional", "concise", "risks", "blockers"],
        },
    ),
    PromptEvalCase(
        name="complex_internal_coordination_slack_jira",
        prompt=(
            "From this meeting, I need a crisp bullet-point summary plus a separate action-items list.\n"
            "Post it to Slack and also create a Jira follow-up.\n"
            "If something is unclear, make reasonable assumptions and add a short 'Open Questions' section."
        ),
        expected={
            "wants_summary": True,
            "communication_channels": ["slack", "jira"],
            "summary_types_any": ["bullet_points", "action_items"],
            "custom_instructions_contains_any": ["open questions", "assumptions"],
        },
    ),
    PromptEvalCase(
        name="complex_discord_telegram_updates",
        prompt=(
            "Summarize this standup with key points and decisions.\n"
            "Send updates to our Discord and Telegram channels.\n"
            "Keep it under 10 bullets and don't include the full transcript."
        ),
        expected={
            "wants_summary": True,
            "wants_transcription": False,
            "communication_channels": ["discord", "telegram"],
            "summary_types_any": ["key_points", "decisions", "bullet_points"],
            "custom_instructions_contains_any": ["under", "bullets"],
        },
    ),
]

SYSTEM_PROMPTS: Dict[str, str] = {
    "default": None,
    "strict_json": (
        "You are a parser. Output ONLY valid JSON matching this schema:\n"
        "{\n"
        "  \"wants_summary\": true|false,\n"
        "  \"wants_transcription\": true|false,\n"
        "  \"communication_channels\": [\"email\"|\"pdf\"|\"slack\"|\"jira\"|\"discord\"|\"telegram\"],\n"
        "  \"recipients\": [\"...\"],\n"
        "  \"summary_types\": [\"standard\"|\"action_items\"|\"decisions\"|\"key_points\"|\"detailed\"|\"bullet_points\"|\"executive\"],\n"
        "  \"custom_instructions\": [\"...\"]\n"
        "}\n"
        "Rules: No markdown. No extra keys. Use empty arrays when unknown."
    ),
}


def _channels_to_values(channels: List[CommunicationChannel]) -> List[str]:
    return [c.value for c in channels]


def _summary_types_to_values(types: List[SummaryType]) -> List[str]:
    return [t.value for t in types]


def _looks_like_email(s: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s.strip()))


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    avg = sum(latencies_ms) / len(latencies_ms)
    p50 = _percentile(latencies_ms, 50) or 0.0
    p95 = _percentile(latencies_ms, 95) or 0.0
    return {"avg": avg, "p50": p50, "p95": p95}


class TestPromptParserEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = "http://localhost:11434"
        cls.fair_latency = os.getenv("PROMPT_EVAL_FAIR_LATENCY", "1").strip().lower() not in {"0", "false", "no"}
        cls.cooldown_ms = int(os.getenv("PROMPT_EVAL_COOLDOWN_MS", "150").strip() or "0")
        cls.models = [
            "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:BF16",
            "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest",
            "hf.co/unsloth/Phi-4-mini-reasoning-GGUF:BF16",
            "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:BF16",
            "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q3_K_XL",
        ]

        cls.system_prompt_keys = ['strict_json', 'default']

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
            try:
                import requests

                requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": " ",
                        "stream": False,
                        "keep_alive": 0,
                        "options": {"temperature": 0, "num_predict": 1},
                    },
                    timeout=30,
                )
            except Exception:
                pass

            gc.collect()
            if self.cooldown_ms > 0:
                time.sleep(self.cooldown_ms / 1000.0)

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
                        stats_by_model[model] = {"model": model, "total": 0, "passed": 0, "failed": 0, "latencies_ms": []}
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
                        stats_by_case[case.name] = {"case": case.name, "total": 0, "passed": 0, "failed": 0, "latencies_ms": []}
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
                        missing = [c for c in exp["communication_channels"] if c not in actual["communication_channels"]]
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
                        non_emails = [r for r in actual["recipients"] if isinstance(r, str) and r.strip() and not _looks_like_email(r)]
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
        rows.append("| model | system_prompt | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
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
        rows.append("| model | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
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
        rows.append("| system_prompt | total | passed | failed | pass_rate | latency_avg_ms | latency_p50_ms | latency_p95_ms |")
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
