from pathlib import Path
from typing import List
import re
import yaml
from .skill import Skill, SkillFrontmatter
from .constants import SKILL_FILENAME, MAX_NAME_LENGTH, MAX_DESCRIPTION_LENGTH, NAME_PATTERN


class SkillDiscovery:
    """
    Discovers and loads skills from files and directories.

    Provides class methods to parse SKILL.md files and extract
    skill metadata. Only frontmatter is parsed at discovery time;
    body content is lazy-loaded when accessed.
    """

    class Error(Exception):
        """Raised when skill discovery or parsing fails."""
        pass

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        """
        Discover a skill from a skill file.

        Parses the YAML frontmatter to extract skill metadata.
        Body content is lazy-loaded when accessed.

        Args:
            path: Path to the SKILL.md file

        Returns:
            Skill object with parsed frontmatter

        Raises:
            SkillDiscovery.Error: If file doesn't exist or is invalid
        """
        if not path.exists():
            raise cls.Error(f"Skill file not found: '{path}'")

        frontmatter = cls._parse_frontmatter(path)
        return Skill(frontmatter=frontmatter, path=path)

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
            SkillDiscovery.Error: If directory doesn't exist or skills are invalid
        """
        if not directory.exists():
            raise cls.Error(f"Skills directory not found: '{directory}'")

        skill_files = list(directory.glob(SKILL_FILENAME))
        skill_files.extend(directory.glob(f"*/{SKILL_FILENAME}"))

        skills = []
        for skill_file in skill_files:
            skill = cls.from_file(skill_file)
            skills.append(skill)

        return skills

    @classmethod
    def _parse_frontmatter(cls, path: Path) -> SkillFrontmatter:
        """
        Parse YAML frontmatter from a skill file.

        Args:
            path: Path to the skill file

        Returns:
            SkillFrontmatter with validated metadata

        Raises:
            SkillDiscovery.Error: If frontmatter is missing or invalid
        """
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise cls.Error(f"Failed to read skill file '{path}': {e}")

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

        return SkillFrontmatter(
            name=name,
            description=description,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata") if isinstance(frontmatter.get("metadata"), dict) else None,
        )
