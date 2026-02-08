# communicator/models.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ChannelType(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    JIRA  = "jira"


@dataclass(frozen=True)
class CommunicationRequest:
    """
    Generic request object for any channel.

    - For email: recipient is an email address.
    - payload is the parsed JSON summary (or any dict).
    """
    recipient: str
    subject: str
    path: str
    metadata: Optional[Dict[str, Any]] = None
