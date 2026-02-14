# communicator/slack_communicator.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import requests
import re

from .interfaces import ICommunicator
from .models import CommunicationRequest

_DIVIDER = "────────────────"


class SlackCommunicator(ICommunicator):
    """
    Slack communicator for sending messages via webhook URLs.
    
    Environment variables:
      - SLACK_WEBHOOK_URL (required): Slack incoming webhook URL
    
    Settings:
      - webhook_url: Slack webhook URL (overrides env var)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        
        if not self.webhook_url:
            raise ValueError(
                "Slack webhook URL is required. Set SLACK_WEBHOOK_URL environment variable "
                "or pass webhook_url parameter"
            )
        
        self.logger.info("SlackCommunicator initialized with webhook")

    def send(self, request: CommunicationRequest) -> Dict[str, Any]:
        """
        Send a message to Slack via webhook.
        
        Args:
            request: Communication request (recipient/channel is ignored for webhooks)
            
        Returns:
            Dict with delivery metadata
        """
        try:
            # Extract markdown content from summary JSON
            markdown_content = self.load_summary_json(request.path)
            
            # Prepare webhook payload
            payload = self._format_webhook_payload(request.subject, markdown_content)
            
            # Send message via webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            response.raise_for_status()
            
            result = {
                "success": True,
                "webhook_url": self.webhook_url.split('/')[-1],  # Show only webhook ID for privacy
                "subject": request.subject,
                "status_code": response.status_code,
                "response_text": response.text
            }
            
            self.logger.info(
                f"Message sent via webhook, status: {response.status_code}"
            )
            
            return result
            
        except requests.exceptions.RequestException as e:
            error_result = {
                "success": False,
                "error": str(e),
                "subject": request.subject,
                "error_type": "request_exception"
            }
            
            self.logger.error(f"Webhook request error: {e}")
            return error_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "subject": request.subject,
                "error_type": "unexpected_error"
            }
            
            self.logger.error(f"Unexpected error sending to Slack: {e}")
            return error_result

    @staticmethod
    def beautify_markdown_for_slack_text(md: str) -> str:
        """
        Best-effort Markdown -> Slack mrkdwn string.
        Designed to avoid the '```markdown``` code block' look and make headings readable.
        """
        if not md or not md.strip():
            return ""

        s = md.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Remove outer fenced code block if user wrapped everything in ```markdown ... ```
        m = re.fullmatch(r"\s*```(?:\w+)?\n([\s\S]*?)\n```\s*", s)
        if m:
            s = m.group(1).strip()

        # Convert ATX headings to Slack-style section titles
        lines = []
        for line in s.split("\n"):
            # Horizontal rules
            if re.match(r"^\s*(---+|\*\*\*+|___+)\s*$", line):
                lines.append(_DIVIDER)
                continue

            # Headings
            h = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if h:
                level = len(h.group(1))
                title = h.group(2).strip()

                # Drop trailing Markdown bold markers inside headings (common in LLM output)
                title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)

                if level == 1:
                    # Prominent top title
                    lines.append(f"*{title}*")
                    lines.append(_DIVIDER)
                elif level == 2:
                    lines.append("")  # spacing
                    lines.append(f"*{title}*")
                else:
                    # Sub-headings: keep compact
                    # Example: "### 1. **Employee Experiences**" -> "*1. Employee Experiences*"
                    title = re.sub(r"^\s*\d+\.\s*", lambda m_: m_.group(0), title)  # no-op but keeps intent clear
                    lines.append(f"*{title}*")
                continue

            lines.append(line.rstrip())

        s = "\n".join(lines)

        # Links: [label](url) -> <url|label>
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"<\2|\1>", s)

        # Bold: **x** / __x__ -> *x*
        s = re.sub(r"\*\*(.+?)\*\*", r"*\1*", s)
        s = re.sub(r"__(.+?)__", r"*\1*", s)

        # Bullets: -, *, + -> •  (preserve indentation)
        s = re.sub(r"(?m)^(\s*)[-*+]\s+", r"\1• ", s)

        # Normalize numbered lists: "1) a" -> "1. a"
        s = re.sub(r"(?m)^(\s*)(\d+)[\.\)]\s*", r"\1\2. ", s)

        # Clean up excessive blank lines
        s = re.sub(r"\n{3,}", "\n\n", s).strip()

        return s

    def _format_webhook_payload(self, subject: str, markdown_content: str) -> dict:
        """
        Format message into Slack webhook payload.
        
        Args:
            subject: Message subject/title
            markdown_content: Markdown content to send
            
        Returns:
            Slack webhook payload dictionary
        """
        # Convert markdown to Slack-friendly format
        text = self.beautify_markdown_for_slack_text(markdown_content)

        # Send as single message
        return {
            "text": subject,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": subject
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text
                    }
                }
            ]
        }

    def _chunk_markdown(self, content: str, max_length: int = 2800) -> list[str]:
        """
        Split markdown content into chunks that fit within Slack limits.
        
        Args:
            content: Markdown content to chunk
            max_length: Maximum length per chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        current_chunk = ""
        
        lines = content.split('\n')
        for line in lines:
            # If adding this line would exceed limit, start new chunk
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = line
                else:
                    # Line itself is too long, force split
                    while len(line) > max_length:
                        chunks.append(line[:max_length])
                        line = line[max_length:]
                    current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk.rstrip())
        
        return chunks

    def load_summary_json(self, json_path: str) -> str:
        p = Path(json_path).expanduser().resolve()
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)['summary_data']['response']

    @staticmethod
    def _render_html(markdown_text: str, title: Optional[str] = None) -> str:
        """
        Render markdown into clean HTML suitable for Slack.
        """
        import markdown

        md = (markdown_text or "").strip()
        body_html = markdown.markdown(
            md,
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html5",
        )

        header = (title or "Meeting Summary").strip()

        # Slack-friendly CSS
        css = """
        <style>
        body { font-family: Arial, sans-serif; line-height: 1.5; color: #111; margin: 0; padding: 0; }
        .page { max-width: 760px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #222; margin-top: 1.5em; margin-bottom: 0.5em; }
        h1 { font-size: 24px; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
        h2 { font-size: 20px; }
        h3 { font-size: 16px; }
        p { margin-bottom: 0.8em; }
        ul, ol { padding-left: 2em; margin-bottom: 0.8em; }
        li { margin-bottom: 0.3em; }
        blockquote { border-left: 4px solid #ddd; padding-left: 1em; margin: 1em 0; color: #555; }
        code { font-family: Consolas, monospace; background: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 1em; border-radius: 4px; overflow-x: auto; }
        hr { border: none; border-top: 1px solid #eee; margin: 2em 0; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
        th, td { border: 1px solid #ddd; padding: 0.5em; text-align: left; }
        th { background: #f9f9f9; }
        </style>
        """

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
{css}
</head>
<body>
<div class="page">
<h1>{header}</h1>
{body_html}
</div>
</body>
</html>
"""

    def cleanup(self) -> None:
        """Clean up resources."""
        # No resources to clean up for webhook implementation
        self.logger.info("SlackCommunicator cleaned up")
