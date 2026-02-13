from __future__ import annotations

import json
import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional
import markdown

from .interfaces import ICommunicator
from .models import CommunicationRequest

import dotenv

dotenv.load_dotenv()


class EmailCommunicator(ICommunicator):
    """
    SMTP-over-SSL email communicator.

    Env vars (defaults shown):
      - SENDER_EMAIL_ACCOUNT (required)
      - SENDER_EMAIL_PASSWORD (required; for Gmail use an App Password)
      - SMTP_SERVER="smtp.gmail.com"
      - SMTP_PORT="465"
    """
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER")
        self.smtp_port = int(smtp_port or os.getenv("SMTP_PORT"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL_ACCOUNT")
        self.sender_password = sender_password or os.getenv("SENDER_EMAIL_PASSWORD")

        if not self.sender_email or not self.sender_password:
            raise ValueError("Missing SENDER_EMAIL_ACCOUNT and/or SENDER_EMAIL_PASSWORD")

    def send(self, request: CommunicationRequest) -> Dict[str, Any]:

        summary = self.load_summary_json(request.path)
        html = self.render_email_html_from_markdown(summary, title=request.subject)

        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(html, "html", "utf-8"))
        msg["Subject"] = request.subject
        msg["From"] = self.sender_email
        msg["To"] = request.recipient

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, request.recipient, msg.as_string())
            self.logger.info("Email sent to %s", request.recipient)
            return {
                "channel": "email",
                "recipient": request.recipient,
                "smtp_server": self.smtp_server,
                "smtp_port": self.smtp_port,
                "status": "sent",
            }
        except Exception as e:
            self.logger.error("Failed to send email: %s", e)
            raise

    def cleanup(self) -> None:
        # SMTP is per-send; nothing to close here
        return

    def load_summary_json(self, json_path: str) -> str:
        p = Path(json_path).expanduser().resolve()
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)['summary_data']['response']

    @staticmethod
    def render_email_html_from_markdown(markdown_text: str, title: Optional[str] = None) -> str:
        """
        Render markdown summary into clean HTML for email.
        - Uses Python-Markdown to convert markdown -> HTML.
        - Adds minimal CSS that works reasonably in most email clients.
        """
        md = (markdown_text or "").strip()
        body_html = markdown.markdown(
            md,
            extensions=[
                "extra",  # tables, fenced code blocks, etc.
                "sane_lists",
                "nl2br",  # keep line breaks
            ],
            output_format="html5",
        )


        return f"""\
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
      </head>
      <body style="font-family: Arial, sans-serif; line-height: 1.5; color: #111;">
        <div style="max-width: 760px; margin: 0 auto; padding: 16px;">
          <div style="font-size: 14px;">
            {body_html}
          </div>

          <hr style="border: none; border-top: 1px solid #eee; margin: 16px 0;" />
          <div style="font-size: 12px; color: #666;">
            Generated automatically.
          </div>
        </div>
      </body>
    </html>
    """