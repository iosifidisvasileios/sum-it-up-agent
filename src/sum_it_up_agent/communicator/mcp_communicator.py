from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Tuple, List

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

from sum_it_up_agent.observability.logger import (
    bind_request_id,
    configure_logging,
    get_logger,
    new_request_id,
)

# Adjust imports to your project layout if needed
from sum_it_up_agent.communicator.factory import CommunicatorFactory, CommunicatorConfig
from sum_it_up_agent.communicator.models import ChannelType, CommunicationRequest

configure_logging()
logger = get_logger("sum_it_up_agent.communicator.mcp")


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return _jsonable(asdict(x))
    if hasattr(x, "value") and isinstance(getattr(x, "value"), str):
        return x.value
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(i) for i in x]
    return x


def _extract_summary_markdown(json_path: str) -> str:
    """
    Your EmailCommunicator expects:
      json['summary_data']['response'] -> markdown string
    """
    p = Path(json_path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return data["summary_data"]["response"]
    except Exception as e:
        raise ValueError("Invalid summary JSON format; expected summary_data.response") from e


class CommunicatorMCP:
    """
    MCP server for the communicator package.

    Tools:
      - send_summary_email: send summary JSON (markdown) to recipient as HTML email.
      - send_summary_slack: send summary JSON (markdown) to Slack channel.
      - export_summary_pdf: export summary JSON (markdown) as a PDF file.
      - render_email_html: render summary markdown to HTML (no sending).
      - cleanup: cleanup cached communicators.

    Env (email):
      - SENDER_EMAIL_ACCOUNT (required unless passed via settings)
      - SENDER_EMAIL_PASSWORD (required unless passed via settings)
      - SMTP_SERVER (optional; defaults to smtp.gmail.com)
      - SMTP_PORT (optional; defaults to 465)

    Env (slack):
      - SLACK_WEBHOOK_URL (required): Slack incoming webhook URL

    PDF export:
      - Requires WeasyPrint. Install with: pip install weasyprint
    
    Slack communication:
      - Uses webhook URLs (no additional dependencies required)
    """

    def __init__(
        self,
        *,
        name: str = "communicator",
        allowed_root: str = None,          # optional sandbox for summary json paths
        serialize_per_instance: bool = True,
        max_cached: int = 8,
    ) -> None:
        self._allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self._serialize = serialize_per_instance
        self._max_cached = max_cached

        # cache key -> (communicator, lock)
        self._cache: dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Tuple[Any, Lock]] = {}

        @lifespan
        async def _ls(_: Any):
            try:
                yield {"server": self}
            finally:
                self._cleanup_all()

        self.mcp = FastMCP(name, lifespan=_ls)
        self._register()

    # -----------------------------
    # path safety
    # -----------------------------
    def _resolve_file(self, p: str) -> str:
        path = Path(p).expanduser().resolve()
        if self._allowed_root is not None:
            try:
                path.relative_to(self._allowed_root)
            except ValueError as e:
                raise ValueError(f"Path outside allowed_root: {path}") from e
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return str(path)

    # -----------------------------
    # communicator cache
    # -----------------------------
    def _cache_key(self, channel: ChannelType, settings: dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        """
        Stable + non-secret key.
        Avoid caching actual passwords. Use len() to disambiguate a bit.
        """
        items: List[Tuple[str, Any]] = []
        for k, v in sorted(settings.items(), key=lambda kv: kv[0]):
            if k in ("sender_password", "SENDER_EMAIL_PASSWORD", "webhook_url", "SLACK_WEBHOOK_URL"):
                items.append((k, f"len:{len(v) if v else 0}"))
            else:
                items.append((k, v))
        return (channel.value, tuple(items))

    def _get_or_create(self, channel: ChannelType, settings: dict[str, Any]) -> Tuple[Any, Lock]:
        st = dict(settings or {})

        # Ensure EmailCommunicator can initialize even if env vars are missing defaults
        if channel == ChannelType.EMAIL:
            st.setdefault("smtp_server", os.getenv("SMTP_SERVER") or "smtp.gmail.com")
            st.setdefault("smtp_port", int(os.getenv("SMTP_PORT") or 465))
            # Prefer env vars if not explicitly provided
            st.setdefault("sender_email", os.getenv("SENDER_EMAIL_ACCOUNT"))
            st.setdefault("sender_password", os.getenv("SENDER_EMAIL_PASSWORD"))

        # Ensure SlackCommunicator can initialize even if env vars are missing defaults
        if channel == ChannelType.SLACK:
            st.setdefault("webhook_url", os.getenv("SLACK_WEBHOOK_URL"))

        key = self._cache_key(channel, st)
        if key in self._cache:
            return self._cache[key]

        if len(self._cache) >= self._max_cached:
            old_key = next(iter(self._cache.keys()))
            comm, _lk = self._cache.pop(old_key)
            try:
                comm.cleanup()
            except Exception:
                pass

        comm = CommunicatorFactory.create(
            CommunicatorConfig(channel=channel, settings=st),
            logger=logger,
        )
        lock = Lock()
        self._cache[key] = (comm, lock)
        return comm, lock

    def _cleanup_all(self) -> None:
        for comm, _lk in self._cache.values():
            try:
                comm.cleanup()
            except Exception:
                pass
        self._cache.clear()

    # -----------------------------
    # MCP API
    # -----------------------------
    def _register(self) -> None:
        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check():
            """Health check endpoint for monitoring."""
            return {"status": "ok", "service": "communicator", "timestamp": time.time()}

        @self.mcp.custom_route("/metrics", methods=["GET"])
        async def prometheus_metrics():
            """Prometheus-compatible metrics endpoint."""
            # Basic metrics for now
            metrics = f"""# HELP communicator_up Status of communicator service
# TYPE communicator_up gauge
communicator_up 1
# HELP communicator_cached_communicators Number of cached communicators
# TYPE communicator_cached_communicators gauge
communicator_cached_communicators {len(self._cache)}
# HELP communicator_start_time_seconds Start time of the service
# TYPE communicator_start_time_seconds gauge
communicator_start_time_seconds {time.time()}
"""
            return metrics

        @self.mcp.resource("comm://channels")
        def channels(_: Context) -> List[str]:
            return [c.value for c in ChannelType]

        @self.mcp.tool
        def send_summary_email(
            recipient: str,
            subject: str,
            summary_json_path: str,
            settings: dict[str, Any] = None,
            uuid: str = None,
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Send the summary JSON (summary_data.response markdown) as an HTML email.

            settings (optional, email):
              - smtp_server, smtp_port
              - sender_email, sender_password
            Prefer env vars for secrets instead of passing sender_password here.
            """
            server: CommunicatorMCP = ctx.lifespan_context["server"]

            correlation_id = uuid or new_request_id()
            with bind_request_id(correlation_id):
                logger.info(
                    "tool_call send_summary_email recipient=%s subject=%s summary_json_path=%s",
                    recipient,
                    subject,
                    summary_json_path,
                )

                comm, lock = server._get_or_create(ChannelType.EMAIL, settings)

                req = CommunicationRequest(
                    recipient=recipient,
                    subject=subject,
                    path=summary_json_path,
                    metadata=None,
                )

                if server._serialize:
                    with lock:
                        result = comm.send(req)
                else:
                    result = comm.send(req)

                logger.info(
                    "tool_result send_summary_email",
                )

                return _jsonable(result)

        @self.mcp.tool
        def export_summary_pdf(
            subject: str,
            summary_json_path: str,
            settings: dict[str, Any] = None,
            uuid: str = None,
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Export the summary JSON (summary_data.response markdown) as a PDF.
            Returns the absolute path to the generated PDF.

            settings (optional, pdf):
              - No settings required for PDF export.
            """
            server: CommunicatorMCP = ctx.lifespan_context["server"]

            correlation_id = uuid or new_request_id()
            with bind_request_id(correlation_id):
                logger.info(
                    "tool_call export_summary_pdf subject=%s summary_json_path=%s",
                    subject,
                    summary_json_path,
                )

                comm, lock = server._get_or_create(ChannelType.PDF, settings)

                req = CommunicationRequest(
                    recipient="",  # Not used for PDF
                    subject=subject,
                    path=summary_json_path,
                    metadata=None,
                )

                if server._serialize:
                    with lock:
                        result = comm.send(req)
                else:
                    result = comm.send(req)

                logger.info(
                    "tool_result export_summary_pdf",
                )

                return _jsonable(result)

        @self.mcp.tool
        def send_summary_slack(
            subject: str,
            summary_json_path: str,
            settings: dict[str, Any] = None,
            uuid: str = None,
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Send the summary JSON (summary_data.response markdown) to Slack via webhook.

            Args:
              subject: Message subject/title
              summary_json_path: Path to summary JSON file
              settings (optional, slack):
                - webhook_url: Slack webhook URL (overrides SLACK_WEBHOOK_URL env var)

            Environment variables:
              - SLACK_WEBHOOK_URL (required): Slack incoming webhook URL

            Note: Webhook URLs are tied to specific channels during setup in Slack.
            """
            server: CommunicatorMCP = ctx.lifespan_context["server"]

            correlation_id = uuid or new_request_id()
            with bind_request_id(correlation_id):
                logger.info(
                    "tool_call send_summary_slack subject=%s summary_json_path=%s",
                    subject,
                    summary_json_path,
                )

                comm, lock = server._get_or_create(ChannelType.SLACK, settings)

                req = CommunicationRequest(
                    recipient="",  # Not used for webhooks
                    subject=subject,
                    path=summary_json_path,
                    metadata=None,
                )

                if server._serialize:
                    with lock:
                        result = comm.send(req)
                else:
                    result = comm.send(req)

                logger.info(
                    "tool_result send_summary_slack",
                )

                return _jsonable(result)

        @self.mcp.tool
        def cleanup(ctx: Context) -> str:
            server: CommunicatorMCP = ctx.lifespan_context["server"]
            server._cleanup_all()
            return "ok"

    # -----------------------------
    # run
    # -----------------------------
    def run(self) -> None:
        transport = os.getenv("MCP_TRANSPORT_COMMUNICATOR").strip().lower()
        if transport == "http":
            host = os.getenv("MCP_HOST_COMMUNICATOR")
            port = int(os.getenv("MCP_PORT_COMMUNICATOR"))
            path = os.getenv("MCP_PATH_COMMUNICATOR")
            self.mcp.run(transport=transport, host=host, port=port, path=path)
        else:
            self.mcp.run()


if __name__ == "__main__":
    server = CommunicatorMCP()
    server.run()
