from typing import List, Optional, Dict
from .skill import Skill
from ..runtime import PythonRuntime


class SkillRegistry:
    """
    Manages skills storage, retrieval, and activation.

    Stores skills by name and provides methods to access and activate them.
    When a skill is activated via activate_skill(), its injection exports (functions
    and variables) are injected into the agent's PythonRuntime.
    """

    def __init__(
        self,
        agent_runtime: Optional[PythonRuntime] = None,
    ):
        """
        Initialize the skill registry.

        Args:
            agent_runtime: Optional PythonRuntime for injecting skill exports
        """
        self._skills: Dict[str, Skill] = {}
        self._agent_runtime = agent_runtime

    def add_skill(self, skill: Skill) -> None:
        """
        Add a skill to the registry.

        Args:
            skill: Skill to register
        """
        self._skills[skill.name] = skill

    def add_skills(self, skills: List[Skill]) -> None:
        """
        Add multiple skills to the registry.

        Args:
            skills: List of skills to register
        """
        for skill in skills:
            self.add_skill(skill)

    def get_skill(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: Name of the skill to retrieve

        Returns:
            Skill if found, None otherwise
        """
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        """
        Get all registered skills.

        Returns:
            List of all skills in the registry
        """
        return list(self._skills.values())

    def describe_skills(self) -> str:
        """
        Generate formatted skill descriptions for system prompt.

        Returns:
            Formatted string with skill names and descriptions
        """
        if not self._skills:
            return "No skills available"

        descriptions = []
        for skill in self._skills.values():
            descriptions.append(f"- {skill.name}: {skill.description}")

        return "\n".join(descriptions)

    def _get_skill_or_raise(self, skill_name: str) -> Skill:
        """Get a skill by name or raise KeyError."""
        skill = self._skills.get(skill_name)
        if not skill:
            available = list(self._skills.keys())
            raise KeyError(f"Skill '{skill_name}' not found. Available skills: {available}")
        return skill

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
        skill = self._get_skill_or_raise(skill_name)

        # Inject skill's exports (functions/variables/types) if available
        if self._agent_runtime and skill.injection:
            for func in skill.injection.functions:
                try:
                    self._agent_runtime.inject_function(func)
                except ValueError:
                    pass  # Function already exists

            for var in skill.injection.variables:
                try:
                    self._agent_runtime.inject_variable(var)
                except ValueError:
                    pass  # Variable already exists

            for type_obj in skill.injection.types:
                try:
                    self._agent_runtime.inject_type(type_obj)
                except ValueError:
                    pass  # Type already exists

        return skill.body_content
