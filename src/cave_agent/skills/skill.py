from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
import importlib.util
from ..runtime import Function, Variable, Type, PythonRuntime
from .constants import INJECTION_FILENAME, SCRIPTS_DIR, REFERENCES_DIR, ASSETS_DIR
from .script import ScriptRunner


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
    _script_runner: Optional[ScriptRunner] = field(default=None, repr=False)

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
    def scripts_dir(self) -> Path:
        """Get the path to the skill's scripts directory."""
        return self.path.parent / SCRIPTS_DIR

    @property
    def has_scripts(self) -> bool:
        """Check if the skill has a scripts directory."""
        return self.scripts_dir.exists() and self.scripts_dir.is_dir()

    @property
    def references_dir(self) -> Path:
        """Get the path to the skill's references directory."""
        return self.path.parent / REFERENCES_DIR

    @property
    def has_references(self) -> bool:
        """Check if the skill has a references directory."""
        return self.references_dir.exists() and self.references_dir.is_dir()

    @property
    def assets_dir(self) -> Path:
        """Get the path to the skill's assets directory."""
        return self.path.parent / ASSETS_DIR

    @property
    def has_assets(self) -> bool:
        """Check if the skill has an assets directory."""
        return self.assets_dir.exists() and self.assets_dir.is_dir()

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

    async def run_script(
        self,
        script_name: str,
        agent_runtime: PythonRuntime,
        **kwargs
    ) -> Any:
        """
        Run a script from the skill's scripts/ directory.

        The script must define a main() function as its entrypoint.
        Scripts run in an isolated IPython environment with access to
        the agent's runtime for retrieving state/data.

        Args:
            script_name: Script filename (e.g., "analyze.py")
            agent_runtime: PythonRuntime for scripts to access agent state/data
            **kwargs: Arguments passed to main()

        Returns:
            Return value from the script's main() function

        Raises:
            FileNotFoundError: If scripts directory or script doesn't exist
            ScriptError: If script execution fails
            RuntimeError: If skill was not activated via activate_skill()
        """
        if self._script_runner is None:
            raise RuntimeError(f"Skill '{self.name}' was not activated. Call activate_skill() first.")

        if not self.has_scripts:
            raise FileNotFoundError(f"Skill '{self.name}' has no scripts/ directory")

        script_path = self.scripts_dir / script_name
        return await self._script_runner.run(script_path, agent_runtime=agent_runtime, **kwargs)

    def read_reference(self, reference_name: str) -> str:
        """
        Read a reference document from the skill's references/ directory.

        Args:
            reference_name: Reference filename (e.g., "GUIDE.md")

        Returns:
            Content of the reference file

        Raises:
            FileNotFoundError: If references directory or file doesn't exist
        """
        if not self.has_references:
            raise FileNotFoundError(f"Skill '{self.name}' has no references/ directory")

        ref_path = self.references_dir / reference_name
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference '{reference_name}' not found in skill '{self.name}'")

        return ref_path.read_text(encoding="utf-8")

    def read_asset(self, asset_name: str) -> bytes:
        """
        Read an asset file from the skill's assets/ directory.

        Args:
            asset_name: Asset filename (e.g., "template.json", "logo.png")

        Returns:
            Raw bytes of the asset file

        Raises:
            FileNotFoundError: If assets directory or file doesn't exist
        """
        if not self.has_assets:
            raise FileNotFoundError(f"Skill '{self.name}' has no assets/ directory")

        asset_path = self.assets_dir / asset_name
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset '{asset_name}' not found in skill '{self.name}'")

        return asset_path.read_bytes()
