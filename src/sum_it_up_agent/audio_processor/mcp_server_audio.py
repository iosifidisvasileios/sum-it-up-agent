from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, List, Optional, Tuple
import time

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

from sum_it_up_agent.observability.logger import (
    bind_request_id,
    configure_logging,
    get_logger,
    new_request_id,
)

# Your library (as used in your example script)
from sum_it_up_agent.audio_processor import AudioProcessingUseCase, ProcessorType, AudioProcessorFactory

configure_logging()
LOGGER = get_logger("sum_it_up_agent.audio_processor.mcp")


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def _predict_output_path(audio_path: str, output_format: str, output_dir: Optional[str]) -> str:
    p = Path(audio_path).expanduser().resolve()
    if output_dir:
        d = Path(output_dir).expanduser().resolve()
        return str(d / f"{p.stem}_transcription.{output_format}")
    return str(p.parent / f"{p.stem}_transcription.{output_format}")


class AudioProcessorMCP:
    """
    MCP server that wraps your existing *use-case + factory* flow.

    - No changes to IAudioProcessor / AudioProcessor / UseCase.
    - Keeps processors warm (cached by preset + overrides) for reuse.
    - Exposes a small MCP surface focused on the agent needs.
    """

    def __init__(
        self,
        *,
        name: str = "audio-processor",
        allowed_root: Optional[str] = None,
        default_hf_token_env: str = "HUGGINGFACE_TOKEN",
        serialize_per_processor: bool = True,
        max_cached_processors: int = 6,
    ) -> None:
        self._allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self._hf_env = default_hf_token_env
        self._serialize = serialize_per_processor
        self._max_cached = max_cached_processors

        # cache key -> (use_case, lock)
        self._cache: dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Tuple[AudioProcessingUseCase, Lock]] = {}

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
    def _resolve_path(self, p: str) -> str:
        path = Path(p).expanduser().resolve()
        if self._allowed_root is not None:
            try:
                path.relative_to(self._allowed_root)
            except ValueError as e:
                raise ValueError(f"Path outside allowed_root: {path}") from e
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return str(path)

    # -----------------------------
    # processor cache
    # -----------------------------
    def _normalize_overrides(self, overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Your factory expects proper types for e.g. device (DeviceType enum).
        Keep this minimal: accept common overrides; everything else passes through.
        """
        if not overrides:
            return {}

        normalized = dict(overrides)

        # Accept device as string; map to your DeviceType by passing through preset config update
        # The safest approach is: use the preset, then set device by reusing preset's type system.
        # Here, we just pass the string and rely on your AudioProcessingConfig validation
        # only if it accepts it. If your AudioProcessingConfig requires DeviceType,
        # pass DeviceType from your library here.
        #
        # If you want strict typing, uncomment and adjust:
        # from src.audio_processor import DeviceType
        # if "device" in normalized and isinstance(normalized["device"], str):
        #     normalized["device"] = DeviceType(normalized["device"].lower())

        return normalized

    def _make_cache_key(self, preset: str, overrides: dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        # stable + hashable
        items = tuple(sorted(overrides.items(), key=lambda kv: kv[0]))
        return (preset, items)

    def _get_or_create_use_case(
        self,
        preset: ProcessorType,
        hf_token: str,
        overrides: Optional[dict[str, Any]],
    ) -> Tuple[AudioProcessingUseCase, Lock]:
        ov = self._normalize_overrides(overrides)
        key = self._make_cache_key(preset.value, ov)

        if key in self._cache:
            return self._cache[key]

        # simple eviction (oldest inserted) to cap GPU memory blowups
        if len(self._cache) >= self._max_cached:
            old_key = next(iter(self._cache.keys()))
            uc, _lk = self._cache.pop(old_key)
            try:
                uc.processor.cleanup()
            except Exception:
                pass

        use_case = AudioProcessingUseCase.create_with_preset(
            processor_type=preset,
            huggingface_token=hf_token,
            config_overrides=ov or None,
            logger=LOGGER,
        )
        lock = Lock()
        self._cache[key] = (use_case, lock)
        return use_case, lock

    def _cleanup_all(self) -> None:
        for (uc, _lk) in self._cache.values():
            try:
                uc.processor.cleanup()
            except Exception:
                pass
        self._cache.clear()

    def _cleanup_use_case(self, use_case: AudioProcessingUseCase) -> None:
        """Clean up a specific use_case and remove it from cache."""
        # Find and remove the use_case from cache
        keys_to_remove = []
        for key, (cached_uc, _lk) in self._cache.items():
            if cached_uc is use_case:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            uc, _lk = self._cache.pop(key)
            try:
                uc.processor.cleanup()
            except Exception:
                pass

    def _cleanup_use_case_async(self, use_case: AudioProcessingUseCase) -> None:
        """Clean up a specific use_case asynchronously in background thread."""
        def cleanup_worker():
            # Find and remove the use_case from cache
            keys_to_remove = []
            for key, (cached_uc, _lk) in self._cache.items():
                if cached_uc is use_case:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                uc, _lk = self._cache.pop(key)
                try:
                    uc.processor.cleanup()
                except Exception:
                    pass
        
        # Run cleanup in background thread
        Thread(target=cleanup_worker, daemon=True).start()

    # -----------------------------
    # MCP registration
    # -----------------------------
    def _register(self) -> None:
        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check():
            """Health check endpoint for monitoring."""
            return {"status": "ok", "service": "audio_processor", "timestamp": time.time()}

        @self.mcp.custom_route("/metrics", methods=["GET"])
        async def prometheus_metrics():
            """Prometheus-compatible metrics endpoint."""
            # Basic metrics for now
            metrics = f"""# HELP audio_processor_up Status of audio processor service
# TYPE audio_processor_up gauge
audio_processor_up 1
# HELP audio_processor_cached_processors Number of cached processors
# TYPE audio_processor_cached_processors gauge
audio_processor_cached_processors {len(self._cache)}
# HELP audio_processor_start_time_seconds Start time of the service
# TYPE audio_processor_start_time_seconds gauge
audio_processor_start_time_seconds {time.time()}
"""
            return metrics

        @self.mcp.resource("audio://presets")
        def presets(_: Context) -> List[str]:
            # Your factory already exposes this mapping
            return list(AudioProcessorFactory.get_available_presets().keys())

        @self.mcp.tool
        def process_audio_file(
            audio_path: str,
            preset: str = "high_quality",
            output_format: str = "json",
            save_to_file: bool = False,
            output_dir: str = None,
            uuid: str = None,
            config_overrides: dict[str, Any] = None,
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Full pipeline via your UseCase:
              - diarization + transcription + merging
              - optional export to json/txt/srt/csv (server-side)
            Returns:
              {segments, summary, saved_path?}
            """
            server: AudioProcessorMCP = ctx.lifespan_context["server"]

            correlation_id = uuid or new_request_id()
            with bind_request_id(correlation_id):
                LOGGER.info(
                    "tool_call process_audio_file audio_path=%s preset=%s output_format=%s save_to_file=%s",
                    audio_path,
                    preset,
                    output_format,
                    save_to_file,
                )

                ap = server._resolve_path(audio_path)

                try:
                    preset_enum = ProcessorType(preset)
                except Exception as e:
                    LOGGER.exception("Invalid preset")
                    raise ValueError(f"Invalid preset='{preset}'. Use audio://presets.") from e

                hf_token = os.getenv(server._hf_env)
                if not hf_token:
                    LOGGER.error("Missing diarization token env=%s", server._hf_env)
                    raise ValueError(f"{server._hf_env} environment variable is required for diarization")

                use_case, lock = server._get_or_create_use_case(preset_enum, hf_token, config_overrides)

                # serialize per processor if requested
                if server._serialize:
                    with lock:
                        segments = use_case.process_audio_file(
                            audio_path=ap,
                            output_format=output_format,
                            save_to_file=save_to_file,
                            output_dir=output_dir,
                        )
                else:
                    segments = use_case.process_audio_file(
                        audio_path=ap,
                        output_format=output_format,
                        save_to_file=save_to_file,
                        output_dir=output_dir,
                    )

                summary = use_case.get_transcription_summary(segments)
                resp: dict[str, Any] = {
                    "audio_path": ap,
                    "preset": preset_enum.value,
                    "segments": _jsonable(segments),
                    "summary": _jsonable(summary),
                }

                if save_to_file:
                    resp["saved_path"] = _predict_output_path(ap, output_format, output_dir)

                server._cleanup_use_case_async(use_case)
                LOGGER.info(
                    "tool_result process_audio_file segments=%s saved_path=%s",
                    len(resp.get("segments") or []),
                    resp.get("saved_path"),
                )
                return resp


        @self.mcp.tool
        def cleanup(ctx: Context) -> str:
            """Drop all cached processors + free resources."""
            server: AudioProcessorMCP = ctx.lifespan_context["server"]
            server._cleanup_all()
            return "ok"

    # -----------------------------
    # run
    # -----------------------------
    def run(self) -> None:
        """
        MCP_TRANSPORT=stdio (default) or http
        MCP_HOST, MCP_PORT, MCP_PATH (for http)
        """
        os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

        transport = os.getenv("MCP_TRANSPORT_AUDIO_PROCESSOR").strip().lower()
        if transport == "http":
            host = os.getenv("MCP_HOST_AUDIO_PROCESSOR")
            port = int(os.getenv("MCP_PORT_AUDIO_PROCESSOR"))
            path = os.getenv("MCP_PATH_AUDIO_PROCESSOR")
            self.mcp.run(transport="http", host=host, port=port, path=path)
        else:
            self.mcp.run()


if __name__ == "__main__":
    server = AudioProcessorMCP()
    server.run()
