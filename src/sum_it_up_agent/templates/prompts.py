from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Type

import importlib.resources


_PROMPT_CACHE: dict[str, str] = {}


def _read_prompt_text(relative_path: str) -> str:
    if relative_path in _PROMPT_CACHE:
        return _PROMPT_CACHE[relative_path]

    text = (
        importlib.resources.files(__package__)
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )

    escaped = text.replace("{", "{{").replace("}", "}}")
    escaped = escaped.replace("{{transcript}}", "{transcript}")

    _PROMPT_CACHE[relative_path] = escaped
    return escaped


class PromptTemplate(ABC):
    """Interface for a meeting summarization prompt template."""
    meeting_type: str

    @abstractmethod
    def render(self, transcript: str) -> str:
        """Return the full prompt string."""
        raise NotImplementedError


# -------------------------
# Concrete template classes
# -------------------------

@dataclass(frozen=True)
class FileBackedPromptTemplate(PromptTemplate):
    meeting_type: str
    prompt_file: str

    def render(self, transcript: str) -> str:
        template = _read_prompt_text(self.prompt_file)
        return template.format(transcript=transcript)


# -------------------------
# Placeholders for others
# -------------------------

@dataclass(frozen=True)
class PlanningCoordinationMeetingTemplate(PromptTemplate):
    meeting_type: str = "planning / coordination meeting"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/planning_coordination_meeting.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class DecisionMakingMeetingTemplate(PromptTemplate):
    meeting_type: str = "decision-making meeting"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/decision_making_meeting.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class BrainstormingSessionTemplate(PromptTemplate):
    meeting_type: str = "brainstorming session"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/brainstorming_session.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class RetrospectivePostmortemTemplate(PromptTemplate):
    meeting_type: str = "retrospective / postmortem"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/retrospective_postmortem.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class TrainingOnboardingTemplate(PromptTemplate):
    meeting_type: str = "training / onboarding"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/training_onboarding.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class InterviewTemplate(PromptTemplate):
    meeting_type: str = "interview"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/interview.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class CustomerCallSalesDemoTemplate(PromptTemplate):
    meeting_type: str = "customer call / sales demo"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/customer_call_sales_demo.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class SupportIncidentCallTemplate(PromptTemplate):
    meeting_type: str = "support / incident call"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/support_incident_call.txt")
        return template.format(transcript=transcript)

@dataclass(frozen=True)
class OtherTemplate(PromptTemplate):
    meeting_type: str = "other"

    def render(self, transcript: str) -> str:
        template = _read_prompt_text("prompts/summarization/other.txt")
        return template.format(transcript=transcript)

# -------------------------
# Factory
# -------------------------

class PromptTemplateFactory:
    _registry: Dict[str, Type[PromptTemplate]] = {}

    @classmethod
    def register(cls, template_cls: Type[PromptTemplate]) -> None:
        meeting_type = getattr(template_cls, "meeting_type", None)
        if not meeting_type:
            raise ValueError(f"{template_cls.__name__} must define meeting_type")
        cls._registry[meeting_type] = template_cls

    @classmethod
    def create(cls, meeting_type: str) -> PromptTemplate:
        if meeting_type not in cls._registry:
            raise KeyError(
                f"Unknown meeting_type: {meeting_type!r}. "
                f"Available: {', '.join(sorted(cls._registry))}"
            )
        return cls._registry[meeting_type]()

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._registry.keys())


# Register defaults (call once at import time)
PromptTemplateFactory.register(
    type(
        "TeamStatusSyncStandupTemplate",
        (FileBackedPromptTemplate,),
        {
            "meeting_type": "team status sync / standup",
            "prompt_file": "prompts/summarization/team_status_sync_standup.txt",
        },
    )
)
PromptTemplateFactory.register(PlanningCoordinationMeetingTemplate)
PromptTemplateFactory.register(DecisionMakingMeetingTemplate)
PromptTemplateFactory.register(BrainstormingSessionTemplate)
PromptTemplateFactory.register(RetrospectivePostmortemTemplate)
PromptTemplateFactory.register(TrainingOnboardingTemplate)
PromptTemplateFactory.register(InterviewTemplate)
PromptTemplateFactory.register(CustomerCallSalesDemoTemplate)
PromptTemplateFactory.register(SupportIncidentCallTemplate)
PromptTemplateFactory.register(OtherTemplate)

# Example:
# tpl = PromptTemplateFactory.create("team status sync / standup")
# prompt = tpl.render(transcript_text)
