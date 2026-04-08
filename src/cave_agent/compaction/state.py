"""Compaction state — tracks health across turns."""

from dataclasses import dataclass


@dataclass
class CompactionState:
    """Mutable state shared across compaction invocations."""
    consecutive_failures: int = 0
