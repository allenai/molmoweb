"""Lightweight memory helpers for logging RSS and triggering GC."""

import gc
import logging
import os
import resource
import sys

logger = logging.getLogger(__name__)


def _rss_current_mb() -> float:
    """Current RSS in MB via /proc/self/statm (Linux) or peak RSS fallback (macOS)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return (pages * resource.getpagesize()) / (1024 * 1024)
    except FileNotFoundError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024


def log_memory(label: str):
    """Log current RSS with a descriptive label."""
    rss = _rss_current_mb()
    logger.info(f"[mem:{label}] RSS={rss:.1f}MB")


def force_gc_and_log(label: str) -> int:
    """Run gc.collect(), log before/after memory, return objects collected."""
    log_memory(f"{label}:before_gc")
    collected = gc.collect()
    log_memory(f"{label}:after_gc")
    logger.info(f"[mem:{label}] gc.collect() freed {collected} objects")
    return collected
