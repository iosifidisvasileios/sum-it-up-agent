# communicator/factory.py
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from .models import ChannelType
from .interfaces import ICommunicator
from .email_communicator import EmailCommunicator
from .pdf_exporter import PDFExporter


@dataclass(frozen=True)
class CommunicatorConfig:
    channel: ChannelType = ChannelType.EMAIL
    settings: Optional[Dict[str, Any]] = None  # channel-specific settings (smtp overrides, etc.)


class CommunicatorFactory:
    @staticmethod
    def create(config: CommunicatorConfig, logger: Optional[logging.Logger] = None) -> ICommunicator:
        settings = config.settings or {}

        if config.channel == ChannelType.EMAIL:
            return EmailCommunicator(
                smtp_server=settings.get("smtp_server"),
                smtp_port=settings.get("smtp_port"),
                sender_email=settings.get("sender_email"),
                sender_password=settings.get("sender_password"),
                logger=logger,
            )

        if config.channel == ChannelType.PDF:
            return PDFExporter(logger=logger)

        raise ValueError(f"Unsupported channel: {config.channel}")
