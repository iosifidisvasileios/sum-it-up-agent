# mcp_topic_classifier_server.py
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, List, Tuple
import dotenv

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

dotenv.load_dotenv()
from sum_it_up_agent.topic_classification import (
    TopicClassificationUseCase,
    TopicClassifierFactory,
    ClassifierType,
    DeviceType,
    EnsembleMethod,
)


# -----------------------------
# Helpers
# -----------------------------

def _jsonable(x: Any) -> Any:
    """Convert dataclasses/enums/etc to JSON-serializable structures."""
    if is_dataclass(x):
        return _jsonable(asdict(x))
    if hasattr(x, "value") and isinstance(getattr(x, "value"), str):
        # Enum-like
        return x.value
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(i) for i in x]
    return x


def _predict_export_path(file_path: str, out_dir: str, fmt: str) -> str:
    p = Path(file_path).expanduser().resolve()
    if out_dir:
        d = Path(out_dir).expanduser().resolve()
        return str(d / f"{p.stem}_topic_classification.{fmt}")
    return str(p.parent / f"{p.stem}_topic_classification.{fmt}")


# -----------------------------
# MCP Server
# -----------------------------

class TopicClassifierMCP:
    """
    MCP wrapper around TopicClassificationUseCase + TopicClassifierFactory.

    - No changes to your existing code.
    - Caches classifiers (preset + overrides) to avoid reloading models.
    - Allows classifying single file, batch lists, or directory globs.
    """

    def __init__(
        self,
        *,
        name: str = "topic-classifier",
        allowed_root: str = None,           # optional safety sandbox for JSON paths
        serialize_per_classifier: bool = True,        # serialize calls per cached classifier
        max_cached_classifiers: int = 6,              # prevent GPU/VRAM blowups
    ) -> None:
        self._allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self._serialize = serialize_per_classifier
        self._max_cached = max_cached_classifiers

        # key -> (use_case, lock)
        self._cache: dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Tuple[TopicClassificationUseCase, Lock]] = {}

        @lifespan
        async def _ls(_: Any):
            try:
                yield {"server": self}
            finally:
                self._cleanup_all()

        self.mcp = FastMCP(name, lifespan=_ls)
        self._register()

    # -----------------------------
    # Path safety
    # -----------------------------
    def _resolve_path(self, p: str) -> str:
        path = Path(p).expanduser().resolve()
        if self._allowed_root is not None:
            try:
                path.relative_to(self._allowed_root)
            except ValueError as e:
                raise ValueError(f"Path outside allowed_root: {path}") from e
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")
        return str(path)

    # -----------------------------
    # Overrides normalization (minimal)
    # -----------------------------
    def _normalize_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize common override fields so your TopicClassificationConfig constructor
        receives the right types.
        """
        if not overrides:
            return {}

        ov = dict(overrides)

        # device can be given as "cpu"|"cuda"|"mps"
        if "device" in ov and isinstance(ov["device"], str):
            ov["device"] = DeviceType(ov["device"].lower())

        # ensemble_method can be given as "mean"|"max"|...
        if "ensemble_method" in ov and isinstance(ov["ensemble_method"], str):
            ov["ensemble_method"] = EnsembleMethod(ov["ensemble_method"].lower())

        # Ensure models/labels are lists if passed as tuples
        if "models" in ov and isinstance(ov["models"], tuple):
            ov["models"] = list(ov["models"])
        if "labels" in ov and isinstance(ov["labels"], tuple):
            ov["labels"] = list(ov["labels"])

        return ov

    def _cache_key(self, preset: str, overrides: dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        """
        Stable + hashable cache key.
        Convert unhashables to repr where needed.
        """
        items: List[Tuple[str, Any]] = []
        for k, v in sorted(overrides.items(), key=lambda kv: kv[0]):
            if isinstance(v, (list, dict)):
                items.append((k, repr(v)))
            elif hasattr(v, "value"):
                items.append((k, getattr(v, "value")))
            else:
                items.append((k, v))
        return (preset, tuple(items))

    def _get_or_create_use_case(
        self,
        preset: ClassifierType,
        overrides: dict[str, Any],
    ) -> Tuple[TopicClassificationUseCase, Lock]:
        ov = self._normalize_overrides(overrides)
        key = self._cache_key(preset.value, ov)

        if key in self._cache:
            return self._cache[key]

        # Evict oldest cached classifier
        if len(self._cache) >= self._max_cached:
            old_key = next(iter(self._cache.keys()))
            uc, _lk = self._cache.pop(old_key)
            try:
                uc.classifier.cleanup()
            except Exception:
                pass

        use_case = TopicClassificationUseCase.create_with_preset(
            classifier_type=preset,
            config_overrides=ov or None,
        )
        lock = Lock()
        self._cache[key] = (use_case, lock)
        return use_case, lock

    def _cleanup_all(self) -> None:
        for uc, _lk in self._cache.values():
            try:
                uc.classifier.cleanup()
            except Exception:
                pass
        self._cache.clear()

    # -----------------------------
    # MCP tools/resources
    # -----------------------------
    def _register(self) -> None:
        @self.mcp.tool("topic://presets")
        def presets() -> List[str]:
            return list(TopicClassifierFactory.get_available_presets().keys())

        @self.mcp.tool
        def classify_conversation_json(
            file_path: str,
            preset: str = "standard",
            config_overrides: dict[str, Any] = None,
            export: bool = False,
            export_format: str = "json",            # json|csv|txt
            export_dir: str = None,
            include_analysis: bool = False,
            ctx: Context = None,
        ) -> dict[str, Any]:
            """
            Classify one transcript JSON file (your full pipeline).
            Optionally export results using your UseCase exporter.
            """
            server: TopicClassifierMCP = ctx.lifespan_context["server"]
            fp = server._resolve_path(file_path)

            try:
                preset_enum = ClassifierType(preset)
            except Exception as e:
                raise ValueError(f"Invalid preset='{preset}'. Use topic://presets.") from e

            use_case, lock = server._get_or_create_use_case(preset_enum, config_overrides)

            if server._serialize:
                with lock:
                    result = use_case.classify_single_file(fp)
            else:
                result = use_case.classify_single_file(fp)

            resp: dict[str, Any] = {
                "file_path": fp,
                "preset": preset_enum.value,
                "result": _jsonable(result),
            }

            if export:
                out_path = _predict_export_path(fp, export_dir, export_format.lower())
                use_case.export_results(
                    [result],
                    out_path,
                    format_type=export_format,
                    include_analysis=include_analysis,
                )
                resp["export_path"] = out_path

            return resp

        @self.mcp.tool
        def cleanup(ctx: Context) -> str:
            server: TopicClassifierMCP = ctx.lifespan_context["server"]
            server._cleanup_all()
            return "ok"

    # -----------------------------
    # Run
    # -----------------------------
    def run(self) -> None:
        """
        Env:
          MCP_TRANSPORT=stdio (default) or http
          MCP_HOST, MCP_PORT, MCP_PATH for http
          TOPIC_ALLOWED_ROOT (optional) for filesystem sandboxing
        """
        transport = os.getenv("MCP_TRANSPORT_TOPIC_CLASSIFIER").strip().lower()
        if transport == "http":
            host = os.getenv("MCP_HOST_TOPIC_CLASSIFIER")
            port = int(os.getenv("MCP_PORT_TOPIC_CLASSIFIER"))
            path = os.getenv("MCP_PATH_TOPIC_CLASSIFIER")
            self.mcp.run(transport=transport, host=host, port=port, path=path)
        else:
            self.mcp.run()


if __name__ == "__main__":
    server = TopicClassifierMCP()
    server.run()
