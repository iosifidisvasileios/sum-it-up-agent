# communicator/pdf_exporter.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces import ICommunicator
from .models import CommunicationRequest

try:
    from weasyprint import HTML, CSS
    _weasyprint_available = True
except Exception:
    _weasyprint_available = False


class PDFExporter(ICommunicator):
    """
    PDF exporter for summaries.
    Uses WeasyPrint to convert HTML (rendered from markdown) to PDF.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        if not _weasyprint_available:
            raise ImportError(
                "WeasyPrint is required for PDF export. Install with: pip install weasyprint"
            )

    def send(self, request: CommunicationRequest) -> Dict[str, Any]:
        """
        Export summary markdown as PDF.
        Returns:
            {"channel": "pdf", "output_path": <path>, "status": "exported"}
        """
        summary_markdown = self.load_summary_json(request.path)
        html = self._render_html(summary_markdown, title=request.subject)

        # Determine output path (same dir as JSON, with .pdf extension)
        json_path = Path(request.path).expanduser().resolve()
        pdf_path = json_path.with_suffix(".pdf")

        # Generate PDF
        HTML(string=html).write_pdf(pdf_path)
        self.logger.info("PDF exported to %s", pdf_path)

        return {
            "channel": "pdf",
            "output_path": str(pdf_path),
            "status": "exported",
        }

    def cleanup(self) -> None:
        # No persistent resources
        return

    def load_summary_json(self, json_path: str) -> str:
        p = Path(json_path).expanduser().resolve()
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)['summary_data']['response']

    @staticmethod
    def _render_html(markdown_text: str, title: Optional[str] = None) -> str:
        """
        Render markdown into clean HTML suitable for PDF.
        """
        import markdown

        md = (markdown_text or "").strip()
        body_html = markdown.markdown(
            md,
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html5",
        )

        header = (title or "Meeting Summary").strip()

        # Minimal print-friendly CSS
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
