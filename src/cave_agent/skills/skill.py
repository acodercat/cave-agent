from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any
import re
import importlib.util
from ..runtime import Function, Variable, Type
from .constants import INJECTION_FILENAME


class SkillInjectionError(Exception):
    """Raised when skill injection setup fails."""
    pass


@dataclass
class SkillFrontmatter:
    """Data extracted from skill file YAML frontmatter."""
    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SkillInjection:
    """Functions, variables, and types to inject from a skill's injection module."""
    functions: List[Function]
    variables: List[Variable]
    types: List[Type]


@dataclass
class Skill:
    """
    Represents a skill with progressive loading.

    Frontmatter (name, description) is loaded at startup from YAML frontmatter.
    The body content and runtime are lazy-loaded on demand when the skill is selected.
    """
    frontmatter: SkillFrontmatter
    path: Path
    _body_content: Optional[str] = field(default=None, repr=False)
    _injection: Optional[SkillInjection] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Get the skill name."""
        return self.frontmatter.name

    @property
    def description(self) -> str:
        """Get the skill description."""
        return self.frontmatter.description

    @property
    def body_content(self) -> str:
        """Get skill body content, loading from file if needed."""
        if self._body_content is None:
            self._parse_body()
        return self._body_content or ""

    @property
    def injection_path(self) -> Path:
        """Get the path to the skill's injection module."""
        return self.path.parent / INJECTION_FILENAME

    @property
    def has_injection(self) -> bool:
        """Check if the skill has an injection module."""
        return self.injection_path.exists()

    @property
    def injection(self) -> Optional[SkillInjection]:
        """Get skill injection exports, loading from file if needed."""
        if self._injection is None and self.has_injection:
            self._setup_injection()
        return self._injection

    def _parse_body(self) -> None:
        """Parse body content from the skill file (everything after frontmatter)."""
        if not self.path.exists():
            self._body_content = ""
            return

        content = self.path.read_text(encoding="utf-8")

        # Remove YAML frontmatter and return the rest
        frontmatter_pattern = r"^---\s*\n.*?\n---\s*\n?"
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        if match:
            self._body_content = content[match.end():].strip()
        else:
            self._body_content = content.strip()

    def _setup_injection(self) -> None:
        """Setup functions and variables from the skill's injection module."""
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(
            f"skill_injection_{self.name}",
            self.injection_path
        )
        if spec is None or spec.loader is None:
            raise SkillInjectionError(f"Failed to load injection module for skill '{self.name}'")

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise SkillInjectionError(f"Error loading injection module for skill '{self.name}': {e}")

        # Get exports from __exports__ list
        if not hasattr(module, "__exports__"):
            raise SkillInjectionError(f"Missing __exports__ in injection module for skill '{self.name}'")

        exports = module.__exports__

        # Collect Function, Variable, and Type instances
        functions = []
        variables = []
        types = []

        for obj in exports:
            if isinstance(obj, Function):
                functions.append(obj)
            elif isinstance(obj, Variable):
                variables.append(obj)
            elif isinstance(obj, Type):
                types.append(obj)

        self._injection = SkillInjection(functions=functions, variables=variables, types=types)
