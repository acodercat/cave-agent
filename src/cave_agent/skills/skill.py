from typing import List, Optional
from ..runtime import Function, Variable, Type


class Skill:
    """
    A skill that can be loaded into a CaveAgent.

    Skills provide domain-specific instructions and optionally inject
    functions, variables, and types into the agent's runtime.

    Users can create skills directly:
        skill = Skill(
            name="my-skill",
            description="Does something useful",
            body_content="# Instructions\\nFollow these steps...",
            functions=[Function(my_func)],
            variables=[Variable("config", value={})],
        )

    Or load from files using SkillDiscovery:
        skill = SkillDiscovery.from_file(Path("my-skill/SKILL.md"))
    """

    name: str
    description: str
    body_content: str
    functions: List[Function]
    variables: List[Variable]
    types: List[Type]

    def __init__(
        self,
        name: str,
        description: str,
        body_content: str = "",
        functions: Optional[List[Function]] = None,
        variables: Optional[List[Variable]] = None,
        types: Optional[List[Type]] = None,
    ):
        """
        Initialize a skill.

        Args:
            name: Skill name (used to identify and activate the skill)
            description: Brief description of what the skill does
            body_content: Instructions and guidance for using the skill
            functions: Functions to inject into runtime when activated
            variables: Variables to inject into runtime when activated
            types: Types to inject into runtime when activated
        """
        self.name = name
        self.description = description
        self.body_content = body_content
        self.functions = functions or []
        self.variables = variables or []
        self.types = types or []

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, description={self.description!r})"
