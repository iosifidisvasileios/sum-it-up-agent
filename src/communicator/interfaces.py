# communicator/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import CommunicationRequest


class ICommunicator(ABC):
    @abstractmethod
    def send(self, request: CommunicationRequest) -> Dict[str, Any]:
        """Send a notification via the channel. Returns channel-specific delivery metadata."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (clients, sessions, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def load_summary_json(self, json_path: str) -> str:
        """Release resources (clients, sessions, etc.)."""
        raise NotImplementedError
