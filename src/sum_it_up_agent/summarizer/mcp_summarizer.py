# mcp_summarizer_server.py
from __future__ import annotations
import dotenv
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, List, Tuple

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

# Your existing library (do not modify it)
from sum_it_up_agent.summarizer import (
    SummarizationUseCase,
    SummarizerType,
    LLMProvider,
)
dotenv.load_dotenv()

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


class SummarizerMCP:
    """
    MCP wrapper around SummarizationUseCase + SummarizerFactory presets.

    - Caches summarizers per (preset + overrides + api_key presence) to avoid re-init.
    - Reads transcript JSON files (audio processor output) and summarizes via templates.
    """

    def __init__(
        self,
        *,
        name: str = "summarizer",
        allowed_root: str = None,
        serialize_per_summarizer: bool = True,
        max_cached_summarizers: int = 6,
    ) -> None:
        self._allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self._serialize = serialize_per_summarizer
        self._max_cached = max_cached_summarizers

        # cache key -> (use_case, lock)
        self._cache: dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Tuple[SummarizationUseCase, Lock]] = {}

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

    def _resolve_out_dir(self, d: str) -> str:
        if d is None:
            return None
        path = Path(d).expanduser().resolve()
        if self._allowed_root is not None:
            try:
                path.relative_to(self._allowed_root)
            except ValueError as e:
                raise ValueError(f"Output dir outside allowed_root: {path}") from e
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # -----------------------------
    # overrides normalization
    # -----------------------------
    def _normalize_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        if not overrides:
            return {}
        ov = dict(overrides)

        # allow llm_provider override as string
        if "llm_provider" in ov and isinstance(ov["llm_provider"], str):
            ov["llm_provider"] = LLMProvider(ov["llm_provider"].lower())

        return ov

    def _cache_key(
        self,
        preset: str,
        overrides: dict[str, Any],
        api_key: str,
    ) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        items: List[Tuple[str, Any]] = []
        for k, v in sorted(overrides.items(), key=lambda kv: kv[0]):
            if isinstance(v, (list, dict)):
                items.append((k, repr(v)))
            elif hasattr(v, "value"):
                items.append((k, getattr(v, "value")))
            else:
                items.append((k, v))

        # do not expose api key; only disambiguate cache if key differs
        if api_key:
            items.append(("__api_key__", f"len:{len(api_key)}"))
        else:
            items.append(("__api_key__", None))

        return (preset, tuple(items))

    def _get_or_create_use_case(
        self,
        preset: SummarizerType,
        overrides: dict[str, Any],
        api_key: str,
    ) -> Tuple[SummarizationUseCase, Lock]:
        ov = self._normalize_overrides(overrides)
        key = self._cache_key(preset.value, ov, api_key)

        if key in self._cache:
            return self._cache[key]

        if len(self._cache) >= self._max_cached:
            old_key = next(iter(self._cache.keys()))
            uc, _lk = self._cache.pop(old_key)
            try:
                uc.summarizer.cleanup()
            except Exception:
                pass

        use_case = SummarizationUseCase.create_with_preset(
            summarizer_type=preset,
            api_key=api_key,
            config_overrides=ov or None,
        )
        lock = Lock()
        self._cache[key] = (use_case, lock)
        return use_case, lock

    def _cleanup_all(self) -> None:
        for uc, _lk in self._cache.values():
            try:
                uc.summarizer.cleanup()
            except Exception:
                pass
        self._cache.clear()

    # -----------------------------
    # MCP endpoints
    # -----------------------------
    def _register(self) -> None:
        @self.mcp.resource("summarizer://presets")
        def presets(_: Context) -> List[str]:
            return [e.value for e in SummarizerType]

        @self.mcp.resource("summarizer://supported_meeting_types")
        def supported_meeting_types(_: Context) -> List[str]:
            # Avoid constructing a summarizer (may require API keys); read from templates.
            try:
                from sum_it_up_agent.templates.prompts import PromptTemplateFactory  # type: ignore
                return list(PromptTemplateFactory.available())
            except Exception:
                return []

        @self.mcp.tool
        def summarize(
            file_path: str,
            meeting_type: str,
            preset: str = "ollama_local",
            user_preferences: list = None,
            config_overrides: dict[str, Any] = None,
            api_key: str = None,
            output_dir: str = None,   # if provided, saves *_summary.json via your UseCase
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Reads a transcript JSON file (audio processor output) and summarizes it.
            meeting_type must match your prompt templates.

            Returns:
              {file_path, preset, meeting_type, result}
            """
            server: SummarizerMCP = ctx.lifespan_context["server"]

            fp = server._resolve_file(file_path)
            out_dir = server._resolve_out_dir(output_dir)

            try:
                preset_enum = SummarizerType(preset)
            except Exception as e:
                raise ValueError(f"Invalid preset='{preset}'. Use summarizer://presets.") from e

            use_case, lock = server._get_or_create_use_case(preset_enum, config_overrides, api_key)

            if server._serialize:
                with lock:
                    result = use_case.summarize_transcription_file(
                        file_path=fp,
                        meeting_type=meeting_type,
                        user_preferences=user_preferences,
                        output_dir=out_dir,
                    )
            else:
                result = use_case.summarize_transcription_file(
                    file_path=fp,
                    meeting_type=meeting_type,
                    user_preferences=user_preferences,
                    output_dir=out_dir,
                )
            # TODO: THIS IS FOR ALL ACTIVE SESSIONS!!! CHANGE IT TO SPECIFIED THREAD
            server._cleanup_all()

            return {
                "file_path": fp,
                "preset": preset_enum.value,
                "meeting_type": meeting_type,
                "result": _jsonable(result),
            }

        @self.mcp.tool
        def cleanup(ctx: Context) -> str:
            server: SummarizerMCP = ctx.lifespan_context["server"]
            server._cleanup_all()
            return "ok"

    # -----------------------------
    # run
    # -----------------------------
    def run(self) -> None:
        transport = os.getenv("MCP_TRANSPORT_SUMMARIZER").strip().lower()
        if transport == "http":
            host = os.getenv("MCP_HOST_SUMMARIZER")
            port = int(os.getenv("MCP_PORT_SUMMARIZER"))
            path = os.getenv("MCP_PATH_SUMMARIZER")
            self.mcp.run(transport=transport, host=host, port=port, path=path)
        else:
            self.mcp.run()


if __name__ == "__main__":
    server = SummarizerMCP()
    server.run()
