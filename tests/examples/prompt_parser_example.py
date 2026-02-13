from __future__ import annotations

import asyncio
import os
from typing import List

import dotenv

from sum_it_up_agent.agent.prompt_parser import PromptParser


dotenv.load_dotenv()


async def run_examples() -> None:
    provider = os.getenv("PROMPT_PARSER_PROVIDER", "ollama")
    model = "lfm2.5-thinking:1.2b"
    base_url = os.getenv("PROMPT_PARSER_BASE_URL", "http://localhost:11434")

    parser = PromptParser(provider=provider, model=model, base_url=base_url, prompt_limit=256)

    prompts: List[str] = [
        # "Summarize meeting.mp3 and email it to me at bill@example.com. Meeting type is standup.",
        # "Please transcribe only (no summary) and export a PDF of the summary for later.",
        # "Create a detailed summary with action items and decisions, and send as PDF.",
        "Just summarize this, and create a pdf report",
    ]

    for i, p in enumerate(prompts, start=1):
        intent = await parser.parse_prompt(p)
        print("=" * 80)
        print(f"Example #{i}")
        print(f"Prompt: {p}")
        print("Parsed intent:")
        print(f"  wants_summary: {intent.wants_summary}")
        print(f"  wants_transcription: {intent.wants_transcription}")
        print(f"  communication_channels: {[c.value for c in intent.communication_channels]}")
        print(f"  recipients: {intent.recipients}")
        print(f"  subject: {intent.subject}")
        print(f"  summary_types: {[s.value for s in intent.summary_types]}")
        print(f"  custom_instructions: {intent.custom_instructions}")


def main() -> None:
    asyncio.run(run_examples())


if __name__ == "__main__":
    main()
