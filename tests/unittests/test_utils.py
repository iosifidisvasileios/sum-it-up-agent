import gc
import re
import time
from typing import Any, Dict, List, Optional

import requests


def _channels_to_values(channels: List[Any]) -> List[str]:
    """Convert communication channels to their string values."""
    return [c.value for c in channels]


def _summary_types_to_values(types: List[Any]) -> List[str]:
    """Convert summary types to their string values."""
    return [t.value for t in types]


def _looks_like_email(s: str) -> bool:
    """Check if a string looks like an email address."""
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s.strip()))


def _percentile(values: List[float], p: float) -> Optional[float]:
    """Calculate the p-th percentile of a list of values."""
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    """Summarize latency statistics."""
    if not latencies_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    avg = sum(latencies_ms) / len(latencies_ms)
    p50 = _percentile(latencies_ms, 50) or 0.0
    p95 = _percentile(latencies_ms, 95) or 0.0
    return {"avg": avg, "p50": p50, "p95": p95}


def perform_fair_latency_cooldown(base_url: str, model: str, cooldown_ms: int) -> None:
    """Perform fair latency cooldown by unloading the model from memory."""
    try:
        requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": " ",
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0, "num_predict": 1},
            },
            timeout=30,
        )
    except Exception:
        pass

    gc.collect()
    if cooldown_ms > 0:
        time.sleep(cooldown_ms / 1000.0)
