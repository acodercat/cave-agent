from typing import Any

from .skill import Skill


class SkillRegistry:
    """Manages skills storage, retrieval, and activation.

    Pure metadata container — does not hold a reference to any runtime.
    Use :meth:`build_skill_store` to produce a dict that can be injected
    into any runtime (IPython or IPyKernel).
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the registry."""
        self._skills[skill.name] = skill

    def add_skills(self, skills: list[Skill]) -> None:
        """Add multiple skills to the registry."""
        for skill in skills:
            self.add_skill(skill)

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name, or None if not found."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def describe_skills(self) -> str:
        """Generate formatted skill descriptions for system prompt."""
        if not self._skills:
            return "No skills available"

        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in self._skills.values()
        )

    def build_skill_store(self) -> dict[str, dict[str, Any]]:
        """Build a skill store for runtime injection.

        Returns::

            {
                "skill-name": {
                    "body_content": "...",
                    "exports": {"func_name": <callable>, "VAR": <value>, ...},
                },
                ...
            }
        """
        store: dict[str, dict[str, Any]] = {}
        for name, skill in self._skills.items():
            exports: dict[str, Any] = {}
            for func in skill.functions:
                exports[func.name] = func.func
            for var in skill.variables:
                exports[var.name] = var.value
            for type_obj in skill.types:
                exports[type_obj.name] = type_obj.value
            store[name] = {
                "body_content": skill.body_content,
                "exports": exports,
            }
        return store
