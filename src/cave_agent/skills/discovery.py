from pathlib import Path
from typing import List
import re
import yaml
import importlib.util
from .skill import Skill
from .constants import SKILL_FILENAME, INJECTION_FILENAME, MAX_NAME_LENGTH, MAX_DESCRIPTION_LENGTH, NAME_PATTERN
from ..runtime import Function, Variable, Type


class SkillDiscovery:
    """
    Discovers and loads skills from files and directories.

    Provides class methods to parse SKILL.md files and optionally
    load injection.py modules to create complete Skill objects.
    """

    class Error(Exception):
        """Raised when skill discovery or parsing fails."""
        pass

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        """
        Load a skill from a SKILL.md file.

        Parses the YAML frontmatter, body content, and optionally
        loads injection.py if present.

        Args:
            path: Path to the SKILL.md file

        Returns:
            Skill object with all content loaded

        Raises:
            SkillDiscovery.Error: If file doesn't exist or is invalid
        """
        if not path.exists():
            raise cls.Error(f"Skill file not found: '{path}'")

        content = cls._read_file(path)
        name, description = cls._parse_frontmatter(path, content)
        body_content = cls._parse_body(content)
        functions, variables, types = cls._load_injection(path.parent, name)

        return Skill(
            name=name,
            description=description,
            body_content=body_content,
            functions=functions,
            variables=variables,
            types=types,
        )

    @classmethod
    def from_directory(cls, directory: Path) -> List[Skill]:
        """
        Discover all skills from a directory.

        Scans for SKILL.md files in the directory root and immediate subdirectories.

        Args:
            directory: Path to the skills directory

        Returns:
            List of discovered Skill objects

        Raises:
            SkillDiscovery.Error: If directory doesn't exist
        """
        if not directory.exists():
            raise cls.Error(f"Skills directory not found: '{directory}'")

        skill_files = list(directory.glob(SKILL_FILENAME))
        skill_files.extend(directory.glob(f"*/{SKILL_FILENAME}"))

        return [cls.from_file(skill_file) for skill_file in skill_files]

    @classmethod
    def _read_file(cls, path: Path) -> str:
        """Read file content."""
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise cls.Error(f"Failed to read skill file '{path}': {e}")

    @classmethod
    def _parse_frontmatter(cls, path: Path, content: str) -> tuple[str, str]:
        """Parse and validate YAML frontmatter, return (name, description)."""
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            raise cls.Error(f"No YAML frontmatter found in '{path}'")

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            raise cls.Error(f"Invalid YAML in '{path}': {e}")

        if not isinstance(frontmatter, dict):
            raise cls.Error(f"Frontmatter must be a dictionary in '{path}'")

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name:
            raise cls.Error(f"Missing required field 'name' in '{path}'")
        if not description:
            raise cls.Error(f"Missing required field 'description' in '{path}'")

        if not isinstance(name, str):
            raise cls.Error(f"Field 'name' must be a string in '{path}'")
        if len(name) > MAX_NAME_LENGTH:
            raise cls.Error(f"Field 'name' exceeds {MAX_NAME_LENGTH} characters in '{path}'")
        if not re.match(NAME_PATTERN, name):
            raise cls.Error(
                f"Field 'name' must contain only lowercase alphanumeric characters and hyphens, "
                f"cannot start/end with hyphen or contain consecutive hyphens in '{path}'"
            )

        if not isinstance(description, str):
            raise cls.Error(f"Field 'description' must be a string in '{path}'")
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise cls.Error(f"Field 'description' exceeds {MAX_DESCRIPTION_LENGTH} characters in '{path}'")

        return name, description

    @classmethod
    def _parse_body(cls, content: str) -> str:
        """Extract body content (everything after frontmatter)."""
        match = re.match(r"^---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
        return content.strip()

    @classmethod
    def _load_injection(cls, skill_dir: Path, skill_name: str) -> tuple[List[Function], List[Variable], List[Type]]:
        """Load injection.py if present, return (functions, variables, types)."""
        injection_path = skill_dir / INJECTION_FILENAME
        if not injection_path.exists():
            return [], [], []

        spec = importlib.util.spec_from_file_location(f"skill_injection_{skill_name}", injection_path)
        if spec is None or spec.loader is None:
            raise cls.Error(f"Failed to load injection module: '{injection_path}'")

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise cls.Error(f"Error loading injection module '{injection_path}': {e}")

        if not hasattr(module, "__exports__"):
            raise cls.Error(f"Missing __exports__ in '{injection_path}'")

        functions = []
        variables = []
        types = []

        for obj in module.__exports__:
            if isinstance(obj, Function):
                functions.append(obj)
            elif isinstance(obj, Variable):
                variables.append(obj)
            elif isinstance(obj, Type):
                types.append(obj)

        return functions, variables, types
