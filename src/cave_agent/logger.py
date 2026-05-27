from enum import IntEnum
from typing import Any

from rich.console import Console
from rich.style import Style
from rich.text import Text


class LogLevel(IntEnum):
    """Log levels for controlling output verbosity."""
    ERROR = 0
    INFO = 1
    DEBUG = 2


class Logger:
    """Structured logger with rich formatting for CaveAgent."""

    _STYLES = {
        LogLevel.DEBUG: Style(color="yellow", bold=False),
        LogLevel.INFO: Style(color="bright_blue", bold=False),
        LogLevel.ERROR: Style(color="bright_red", bold=True),
    }

    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.console = Console()
        self.level = level

    def _log(self, title: str, content: Any, style: str, level: LogLevel):
        if level <= self.level:
            message = Text()
            message.append(f"[{level.name}] ", self._STYLES[level])
            parsed_style = Style.parse(style)
            message.append(f"{title}: \n", parsed_style)
            message.append(str(content), parsed_style)
            self.console.print(message)

    def debug(self, title: str, content: Any, style: str = "yellow"):
        self._log(title, content, style, LogLevel.DEBUG)

    def info(self, title: str, content: Any, style: str = "blue"):
        self._log(title, content, style, LogLevel.INFO)

    def error(self, title: str, content: Any, style: str = "red"):
        self._log(title, content, style, LogLevel.ERROR)
