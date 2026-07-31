"""Compaction circuit-breaker state."""

from dataclasses import dataclass

MAX_CONSECUTIVE_FAILURES = 3

# Summarization attempts skipped after the breaker trips, before one is let
# through again. Counted in attempts rather than wall-clock or agent steps so
# the state is self-contained: an earlier design took the step number from the
# caller, and the one production call site never passed it, leaving the breaker
# latched open for the lifetime of the agent.
COOLDOWN_ATTEMPTS = 10


@dataclass
class CompactionState:
    """Health of the LLM summarization tier, scoped to one agent session."""

    consecutive_failures: int = 0
    cooldown_remaining: int = 0

    @property
    def is_open(self) -> bool:
        """Whether the breaker is currently tripped (pure; does not tick)."""
        return self.cooldown_remaining > 0

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.cooldown_remaining = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.cooldown_remaining = COOLDOWN_ATTEMPTS

    def should_skip(self) -> bool:
        """Whether to skip summarization, consuming one cooldown tick.

        Called once per summarization attempt. While the breaker is open this
        returns ``True`` and counts down; on the final tick it also clears the
        failure count, so the next attempt gets a genuine retry rather than
        tripping again on the first stale failure.
        """
        if self.cooldown_remaining <= 0:
            return False
        self.cooldown_remaining -= 1
        if self.cooldown_remaining == 0:
            self.consecutive_failures = 0
        return True
