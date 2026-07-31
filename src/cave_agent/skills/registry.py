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

        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())

    def build_skill_store(self) -> dict[str, str]:
        """Build the instruction store consumed by ``activate_skill``.

        Returns::

            {
                "skill-name": "skill instructions",
                ...
            }

        Runtime exports are deliberately absent. ``CaveAgent`` registers them
        as managed hidden bindings so reset and collision semantics stay under
        runtime ownership.
        """
        return {name: skill.body_content for name, skill in self._skills.items()}
