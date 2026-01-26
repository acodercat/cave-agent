from typing import List, Dict, Optional
from .skill import Skill
from ..runtime import PythonRuntime


class SkillRegistry:
    """
    Manages skills storage, retrieval, and activation.

    Stores skills by name and provides methods to access and activate them.
    When a skill is activated via activate_skill(), its functions, variables,
    and types are injected into the agent's PythonRuntime.
    """

    def __init__(self, agent_runtime: PythonRuntime):
        """
        Initialize the skill registry.

        Args:
            agent_runtime: PythonRuntime for injecting skill exports
        """
        self._skills: Dict[str, Skill] = {}
        self._agent_runtime = agent_runtime

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the registry."""
        self._skills[skill.name] = skill

    def add_skills(self, skills: List[Skill]) -> None:
        """Add multiple skills to the registry."""
        for skill in skills:
            self.add_skill(skill)

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name, or None if not found."""
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
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

    def activate_skill(self, skill_name: str) -> str:
        """
        Activate a skill and return its instructions.

        Call this function ONCE when you need specialized guidance for a task.
        Print the returned value to see the skill's instructions, then follow
        them to complete the task. Do NOT call again for the same skill.

        Args:
            skill_name: The exact name of the skill to activate (from the skills list)

        Returns:
            The skill's instructions and guidance

        Raises:
            KeyError: If skill is not found
        """
        skill = self._skills.get(skill_name)
        if not skill:
            available = list(self._skills.keys())
            raise KeyError(f"Skill '{skill_name}' not found. Available skills: {available}")

        # Inject skill's functions, variables, and types into runtime
        for func in skill.functions:
            try:
                self._agent_runtime.inject_function(func)
            except ValueError:
                pass  # Already exists

        for var in skill.variables:
            try:
                self._agent_runtime.inject_variable(var)
            except ValueError:
                pass  # Already exists

        for type_obj in skill.types:
            try:
                self._agent_runtime.inject_type(type_obj)
            except ValueError:
                pass  # Already exists

        return skill.body_content
