import pytest
import tempfile
from pathlib import Path

from cave_agent.skills import (
    Skill, SkillFrontmatter, SkillInjection, SkillInjectionError,
    SkillRegistry, SkillDiscovery
)
from cave_agent import CaveAgent
from cave_agent.runtime import PythonRuntime, Function, Variable, Type
from cave_agent.models import Model


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_skill_content():
    """Valid SKILL.md content."""
    return """---
name: test-skill
description: A test skill for unit testing
---

# Test Skill Instructions

Follow these steps:
1. Step one
2. Step two
"""


@pytest.fixture
def skill_file(temp_dir, valid_skill_content):
    """Create a valid skill file."""
    path = temp_dir / "SKILL.md"
    path.write_text(valid_skill_content)
    return path


@pytest.fixture
def mock_model():
    """Mock model for agent tests."""
    class MockModel(Model):
        async def call(self, messages):
            pass
        async def stream(self, messages):
            yield ""
    return MockModel()


# =============================================================================
# SkillFrontmatter Tests
# =============================================================================

class TestSkillFrontmatter:
    def test_create_metadata(self):
        """Test creating SkillFrontmatter."""
        metadata = SkillFrontmatter(name="test", description="A test skill")
        assert metadata.name == "test"
        assert metadata.description == "A test skill"


# =============================================================================
# SkillInjection Tests
# =============================================================================

class TestSkillInjection:
    def test_skill_injection_dataclass(self):
        """Test SkillInjection structure with functions, variables, types."""
        def sample_func():
            pass

        func = Function(sample_func, description="A sample function")
        var = Variable("sample_var", value=42, description="A sample variable")
        type_obj = Type(int, description="Integer type")

        injection = SkillInjection(
            functions=[func],
            variables=[var],
            types=[type_obj]
        )

        assert len(injection.functions) == 1
        assert len(injection.variables) == 1
        assert len(injection.types) == 1
        assert injection.functions[0].name == "sample_func"
        assert injection.variables[0].name == "sample_var"
        assert injection.types[0].name == "int"

    def test_skill_injection_empty(self):
        """Test SkillInjection with empty lists."""
        injection = SkillInjection(functions=[], variables=[], types=[])

        assert injection.functions == []
        assert injection.variables == []
        assert injection.types == []


class TestSkillHasInjection:
    def test_has_injection_true(self, temp_dir):
        """Test has_injection returns True when injection.py exists."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function

def helper():
    pass

__exports__ = [Function(helper)]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        assert skill.has_injection is True

    def test_has_injection_false(self, skill_file):
        """Test has_injection returns False when no injection.py."""
        skill = SkillDiscovery.from_file(skill_file)
        assert skill.has_injection is False

    def test_injection_path(self, temp_dir):
        """Test injection_path points to correct location."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")

        expected_path = temp_dir / "injection.py"
        assert skill.injection_path == expected_path


class TestSkillInjectionLoading:
    def test_injection_lazy_loading(self, temp_dir):
        """Test _injection is None before access."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function

def helper():
    pass

__exports__ = [Function(helper)]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        assert skill._injection is None

        # Access triggers loading
        _ = skill.injection
        assert skill._injection is not None

    def test_injection_loads_functions(self, temp_dir):
        """Test injection loads Function objects from __exports__."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function

def process_data(data):
    return data * 2

def analyze_results(results):
    return len(results)

__exports__ = [
    Function(process_data, description="Process data"),
    Function(analyze_results, description="Analyze results"),
]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        injection = skill.injection

        assert injection is not None
        assert len(injection.functions) == 2
        assert injection.functions[0].name == "process_data"
        assert injection.functions[1].name == "analyze_results"

    def test_injection_loads_variables(self, temp_dir):
        """Test injection loads Variable objects from __exports__."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Variable

CONFIG = {"key": "value", "threshold": 0.5}
MAX_RETRIES = 3

__exports__ = [
    Variable("CONFIG", value=CONFIG, description="Configuration dict"),
    Variable("MAX_RETRIES", value=MAX_RETRIES, description="Max retry count"),
]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        injection = skill.injection

        assert injection is not None
        assert len(injection.variables) == 2
        assert injection.variables[0].name == "CONFIG"
        assert injection.variables[0].value == {"key": "value", "threshold": 0.5}
        assert injection.variables[1].name == "MAX_RETRIES"
        assert injection.variables[1].value == 3

    def test_injection_loads_types(self, temp_dir):
        """Test injection loads Type objects from __exports__."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Type
from dataclasses import dataclass

@dataclass
class DataPoint:
    x: float
    y: float

class Processor:
    def process(self, data):
        return data

__exports__ = [
    Type(DataPoint, description="A data point"),
    Type(Processor, description="Data processor"),
]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        injection = skill.injection

        assert injection is not None
        assert len(injection.types) == 2
        assert injection.types[0].name == "DataPoint"
        assert injection.types[1].name == "Processor"

    def test_injection_loads_mixed_exports(self, temp_dir):
        """Test injection correctly categorizes mixed exports."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function, Variable, Type

def helper():
    pass

VALUE = 42

class MyClass:
    pass

__exports__ = [
    Function(helper),
    Variable("VALUE", value=VALUE),
    Type(MyClass),
]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        injection = skill.injection

        assert injection is not None
        assert len(injection.functions) == 1
        assert len(injection.variables) == 1
        assert len(injection.types) == 1
        assert injection.functions[0].name == "helper"
        assert injection.variables[0].name == "VALUE"
        assert injection.types[0].name == "MyClass"

    def test_injection_missing_exports_raises_error(self, temp_dir):
        """Test missing __exports__ raises SkillInjectionError."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
def helper():
    pass
# No __exports__ defined
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")

        with pytest.raises(SkillInjectionError) as exc_info:
            _ = skill.injection

        assert "Missing __exports__" in str(exc_info.value)

    def test_injection_module_load_error(self, temp_dir):
        """Test invalid module raises SkillInjectionError."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
# Invalid Python syntax
def broken(
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")

        with pytest.raises(SkillInjectionError) as exc_info:
            _ = skill.injection

        assert "Error loading injection module" in str(exc_info.value)

    def test_injection_returns_none_when_no_file(self, skill_file):
        """Test injection returns None when no injection.py exists."""
        skill = SkillDiscovery.from_file(skill_file)

        assert skill.injection is None


# =============================================================================
# Skill Tests
# =============================================================================

class TestSkill:
    def test_skill_properties(self, skill_file):
        """Test Skill name and description properties."""
        frontmatter = SkillFrontmatter(name="test-skill", description="A test skill")
        skill = Skill(frontmatter=frontmatter, path=skill_file)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"

    def test_skill_body_content_lazy_loading(self, skill_file):
        """Test that body_content is lazy loaded."""
        frontmatter = SkillFrontmatter(name="test-skill", description="A test skill")
        skill = Skill(frontmatter=frontmatter, path=skill_file)

        # _body_content should be None before accessing
        assert skill._body_content is None

        # Access body_content triggers loading
        body = skill.body_content
        assert skill._body_content is not None
        assert "Test Skill Instructions" in body

    def test_skill_body_content_strips_frontmatter(self, skill_file):
        """Test that body_content doesn't include YAML frontmatter."""
        frontmatter = SkillFrontmatter(name="test-skill", description="A test skill")
        skill = Skill(frontmatter=frontmatter, path=skill_file)

        body = skill.body_content
        assert "---" not in body
        assert "name:" not in body
        assert "description:" not in body

    def test_skill_nonexistent_file(self, temp_dir):
        """Test skill with nonexistent file returns empty body_content."""
        frontmatter = SkillFrontmatter(name="missing", description="Missing skill")
        skill = Skill(frontmatter=frontmatter, path=temp_dir / "nonexistent.md")

        assert skill.body_content == ""


# =============================================================================
# SkillDiscovery Tests
# =============================================================================

class TestSkillDiscoveryFromFile:
    def test_parse_valid_skill(self, skill_file):
        """Test loading a valid skill file."""
        skill = SkillDiscovery.from_file(skill_file)

        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"

    def test_parse_missing_name(self, temp_dir):
        """Test that missing name raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\ndescription: No name\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'name'" in str(exc_info.value)

    def test_parse_missing_description(self, temp_dir):
        """Test that missing description raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: no-desc\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'description'" in str(exc_info.value)

    def test_parse_empty_name(self, temp_dir):
        """Test that empty name raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: \"\"\ndescription: Has desc\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'name'" in str(exc_info.value)

    def test_parse_empty_description(self, temp_dir):
        """Test that empty description raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: has-name\ndescription: \"\"\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'description'" in str(exc_info.value)

    def test_parse_no_frontmatter(self, temp_dir):
        """Test that missing frontmatter raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("Just content, no frontmatter")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "No YAML frontmatter found" in str(exc_info.value)

    def test_parse_invalid_yaml(self, temp_dir):
        """Test that invalid YAML raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: [invalid yaml\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Invalid YAML" in str(exc_info.value)

    def test_parse_nonexistent_file(self, temp_dir):
        """Test that nonexistent file raises SkillDiscovery.Error."""
        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(temp_dir / "nonexistent.md")
        assert "Skill file not found" in str(exc_info.value)


class TestSkillDiscoveryNameValidation:
    def test_name_too_long_rejected(self, temp_dir):
        """Test that name exceeding 64 chars raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        long_name = "a" * 65
        path.write_text(f"---\nname: {long_name}\ndescription: Desc\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "exceeds 64 characters" in str(exc_info.value)

    def test_name_max_length_accepted(self, temp_dir):
        """Test that name with exactly 64 chars is accepted."""
        path = temp_dir / "SKILL.md"
        max_name = "a" * 64
        path.write_text(f"---\nname: {max_name}\ndescription: Desc\n---\nContent")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None
        assert skill.name == max_name


class TestSkillDiscoveryDescriptionValidation:
    def test_description_too_long_rejected(self, temp_dir):
        """Test that description exceeding 1024 chars raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        long_desc = "a" * 1025
        path.write_text(f"---\nname: test\ndescription: {long_desc}\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "exceeds 1024 characters" in str(exc_info.value)

    def test_description_max_length_accepted(self, temp_dir):
        """Test that description with exactly 1024 chars is accepted."""
        path = temp_dir / "SKILL.md"
        max_desc = "a" * 1024
        path.write_text(f"---\nname: test\ndescription: {max_desc}\n---\nContent")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None


class TestSkillDiscoveryOptionalFields:
    def test_parse_license(self, temp_dir):
        """Test parsing license field."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: test\ndescription: Desc\nlicense: MIT\n---\nContent")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None
        assert skill.frontmatter.license == "MIT"

    def test_parse_compatibility(self, temp_dir):
        """Test parsing compatibility field."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: test\ndescription: Desc\ncompatibility: Requires Python 3.10+\n---\nContent")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None
        assert skill.frontmatter.compatibility == "Requires Python 3.10+"

    def test_parse_metadata(self, temp_dir):
        """Test parsing metadata field."""
        path = temp_dir / "SKILL.md"
        path.write_text("""---
name: test
description: Desc
metadata:
  author: test-org
  version: "1.0"
---
Content""")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None
        assert skill.frontmatter.metadata == {"author": "test-org", "version": "1.0"}

    def test_all_optional_fields(self, temp_dir):
        """Test parsing all optional fields together."""
        path = temp_dir / "SKILL.md"
        path.write_text("""---
name: pdf-processing
description: Extract text and tables from PDF files.
license: Apache-2.0
compatibility: Requires pdfplumber package
metadata:
  author: example-org
  version: "1.0"
---
Content""")

        skill = SkillDiscovery.from_file(path)
        assert skill is not None
        assert skill.name == "pdf-processing"
        assert skill.frontmatter.license == "Apache-2.0"
        assert skill.frontmatter.compatibility == "Requires pdfplumber package"
        assert skill.frontmatter.metadata == {"author": "example-org", "version": "1.0"}


class TestSkillDiscoveryFromDirectory:
    def test_from_directory(self, temp_dir):
        """Test discovering skills from directory."""
        # Create skill in root
        (temp_dir / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: Root skill\n---\nRoot content"
        )

        # Create skill in subdirectory
        subdir = temp_dir / "subskill"
        subdir.mkdir()
        (subdir / "SKILL.md").write_text(
            "---\nname: sub-skill\ndescription: Sub skill\n---\nSub content"
        )

        skills = SkillDiscovery.from_directory(temp_dir)

        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "root-skill" in names
        assert "sub-skill" in names

    def test_from_empty_directory(self, temp_dir):
        """Test discovering from empty directory returns empty list."""
        skills = SkillDiscovery.from_directory(temp_dir)
        assert skills == []

    def test_from_nonexistent_directory(self):
        """Test discovering from nonexistent directory raises SkillDiscovery.Error."""
        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_directory(Path("/nonexistent/path"))
        assert "Skills directory not found" in str(exc_info.value)


# =============================================================================
# SkillRegistry Tests
# =============================================================================

class TestSkillRegistryManagement:
    def test_add_skill(self, skill_file):
        """Test adding a skill to registry."""
        registry = SkillRegistry()
        skill = SkillDiscovery.from_file(skill_file)

        registry.add_skill(skill)

        assert registry.get_skill("test-skill") is skill

    def test_add_skills(self, temp_dir):
        """Test adding multiple skills."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: skill1\ndescription: First\n---\nContent 1"
        )
        subdir = temp_dir / "sub"
        subdir.mkdir()
        (subdir / "SKILL.md").write_text(
            "---\nname: skill2\ndescription: Second\n---\nContent 2"
        )

        registry = SkillRegistry()
        skills = SkillDiscovery.from_directory(temp_dir)
        registry.add_skills(skills)

        assert len(registry.list_skills()) == 2

    def test_get_skill_not_found(self):
        """Test getting nonexistent skill returns None."""
        registry = SkillRegistry()
        assert registry.get_skill("nonexistent") is None

    def test_list_skills(self, skill_file):
        """Test listing all skills."""
        registry = SkillRegistry()
        skill = SkillDiscovery.from_file(skill_file)
        registry.add_skill(skill)

        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_describe_skills(self, skill_file):
        """Test skill descriptions for system prompt."""
        registry = SkillRegistry()
        skill = SkillDiscovery.from_file(skill_file)
        registry.add_skill(skill)

        description = registry.describe_skills()
        assert "test-skill" in description
        assert "A test skill for unit testing" in description

    def test_describe_skills_empty(self):
        """Test empty registry returns 'No skills available'."""
        registry = SkillRegistry()
        assert registry.describe_skills() == "No skills available"


class TestSkillRegistryActivateSkill:
    def test_activate_skill(self, skill_file):
        """Test activate_skill returns prompt content."""
        registry = SkillRegistry()
        skill = SkillDiscovery.from_file(skill_file)
        registry.add_skill(skill)

        prompt = registry.activate_skill("test-skill")
        assert "Test Skill Instructions" in prompt

    def test_activate_skill_not_found(self):
        """Test activate_skill raises KeyError for unknown skill."""
        registry = SkillRegistry()

        with pytest.raises(KeyError) as exc_info:
            registry.activate_skill("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "Available skills" in str(exc_info.value)


class TestActivateSkillInjection:
    def test_activate_skill_injects_functions(self, temp_dir):
        """Test activate_skill injects functions into runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function

def skill_helper(x):
    return x * 2

__exports__ = [Function(skill_helper, description="Double the input")]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("test")

        functions = runtime.describe_functions()
        assert "skill_helper" in functions

    def test_activate_skill_injects_variables(self, temp_dir):
        """Test activate_skill injects variables into runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Variable

SKILL_CONFIG = {"threshold": 0.5}

__exports__ = [Variable("SKILL_CONFIG", value=SKILL_CONFIG)]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("test")

        variables = runtime.describe_variables()
        assert "SKILL_CONFIG" in variables

    def test_activate_skill_injects_types(self, temp_dir):
        """Test activate_skill injects types into runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Type
from dataclasses import dataclass

@dataclass
class SkillResult:
    value: int
    status: str

__exports__ = [Type(SkillResult, description="Result from skill")]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("test")

        types = runtime.describe_types()
        assert "SkillResult" in types

    @pytest.mark.asyncio
    async def test_injected_type_usable_in_runtime(self, temp_dir):
        """Test injected type can be instantiated in runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Type
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

__exports__ = [Type(Point)]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("test")

        result = await runtime.execute("p = Point(1.0, 2.0)")
        assert result.success

        result = await runtime.execute("print(p.x, p.y)")
        assert "1.0 2.0" in result.stdout

    def test_duplicate_injection_ignored(self, temp_dir):
        """Test duplicate injection doesn't raise error."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function

def helper():
    return 42

__exports__ = [Function(helper)]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        # First activation
        registry.activate_skill("test")

        # Second activation should not raise
        registry.activate_skill("test")

        functions = runtime.describe_functions()
        assert "helper" in functions


class TestActivateSkillCombinedInjection:
    def test_activate_skill_injects_all_types(self, temp_dir):
        """Test activate_skill injects functions, variables, and types together."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: combined\ndescription: Combined test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function, Variable, Type
from dataclasses import dataclass

def process(data):
    return data

CONFIG = {"enabled": True}

@dataclass
class Result:
    value: int

__exports__ = [
    Function(process),
    Variable("CONFIG", value=CONFIG),
    Type(Result),
]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("combined")

        assert "process" in runtime.describe_functions()
        assert "CONFIG" in runtime.describe_variables()
        assert "Result" in runtime.describe_types()

    @pytest.mark.asyncio
    async def test_all_injected_items_accessible(self, temp_dir):
        """Test all injected items are accessible in executed code."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function, Variable, Type
from dataclasses import dataclass

def multiply(x, y):
    return x * y

FACTOR = 10

@dataclass
class Data:
    value: int

__exports__ = [
    Function(multiply),
    Variable("FACTOR", value=FACTOR),
    Type(Data),
]
""")

        runtime = PythonRuntime()
        registry = SkillRegistry(agent_runtime=runtime)
        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")
        registry.add_skill(skill)

        registry.activate_skill("test")

        # Test function
        result = await runtime.execute("result = multiply(3, 4)")
        assert result.success
        result = await runtime.execute("print(result)")
        assert "12" in result.stdout

        # Test variable
        result = await runtime.execute("print(FACTOR)")
        assert "10" in result.stdout

        # Test type
        result = await runtime.execute("d = Data(value=42)")
        assert result.success
        result = await runtime.execute("print(d.value)")
        assert "42" in result.stdout


# =============================================================================
# CaveAgent Skills Integration Tests
# =============================================================================

class TestAgentSkillsInit:
    def test_agent_with_skills_dir(self, temp_dir, mock_model):
        """Test agent initialization with skills_dir."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test skill\n---\nContent"
        )

        agent = CaveAgent(model=mock_model, skills_dir=temp_dir)

        assert agent._skill_registry.get_skill("test") is not None

    def test_agent_with_skills_list(self, skill_file, mock_model):
        """Test agent initialization with skills list."""
        skill = SkillDiscovery.from_file(skill_file)

        agent = CaveAgent(model=mock_model, skills=[skill])

        assert agent._skill_registry.get_skill("test-skill") is not None

    def test_agent_with_both_skills_sources(self, temp_dir, mock_model):
        """Test agent with both skills_dir and skills list."""
        # Skill in directory
        (temp_dir / "SKILL.md").write_text(
            "---\nname: dir-skill\ndescription: From dir\n---\nDir content"
        )

        # Skill from list
        subdir = temp_dir / "other"
        subdir.mkdir()
        (subdir / "SKILL.md").write_text(
            "---\nname: list-skill\ndescription: From list\n---\nList content"
        )
        list_skill = SkillDiscovery.from_file(subdir / "SKILL.md")

        agent = CaveAgent(
            model=mock_model,
            skills_dir=temp_dir,
            skills=[list_skill]
        )

        assert agent._skill_registry.get_skill("dir-skill") is not None
        assert agent._skill_registry.get_skill("list-skill") is not None

    def test_agent_no_skills(self, mock_model):
        """Test agent without skills."""
        agent = CaveAgent(model=mock_model)

        assert agent._skill_registry.list_skills() == []


class TestAgentSkillsSystemPrompt:
    def test_skills_in_system_prompt(self, temp_dir, mock_model):
        """Test skills appear in system prompt."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: My description\n---\nContent"
        )

        agent = CaveAgent(model=mock_model, skills_dir=temp_dir)
        prompt = agent.build_system_prompt()

        assert "<skills>" in prompt
        assert "my-skill" in prompt
        assert "My description" in prompt

    def test_no_skills_in_system_prompt(self, mock_model):
        """Test system prompt with no skills."""
        agent = CaveAgent(model=mock_model)
        prompt = agent.build_system_prompt()

        assert "<skills>" in prompt
        assert "No skills available" in prompt


class TestAgentSkillsRuntime:
    def test_activate_skill_function_injected(self, temp_dir, mock_model):
        """Test activate_skill function is injected into runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )

        agent = CaveAgent(model=mock_model, skills_dir=temp_dir)

        functions = agent.runtime.describe_functions()
        assert "activate_skill" in functions

    def test_activate_skill_not_injected_without_skills(self, mock_model):
        """Test activate_skill is not injected when no skills."""
        agent = CaveAgent(model=mock_model)

        functions = agent.runtime.describe_functions()
        assert "activate_skill" not in functions

    @pytest.mark.asyncio
    async def test_activate_skill_execution(self, temp_dir, mock_model):
        """Test executing activate_skill in runtime."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test\n---\nSkill instructions here"
        )

        agent = CaveAgent(model=mock_model, skills_dir=temp_dir)

        result = await agent.runtime.execute('output = activate_skill("my-skill")')
        assert result.success

        result = await agent.runtime.execute('print(output)')
        assert "Skill instructions here" in result.stdout

    @pytest.mark.asyncio
    async def test_activate_skill_execution_invalid_name(self, temp_dir, mock_model):
        """Test activate_skill with invalid name raises error."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: valid\ndescription: Test\n---\nContent"
        )

        agent = CaveAgent(model=mock_model, skills_dir=temp_dir)

        result = await agent.runtime.execute('activate_skill("invalid")')
        assert not result.success
        assert result.error is not None


# =============================================================================
# Function.is_async Tests
# =============================================================================

class TestFunctionIsAsync:
    def test_sync_function_is_async_false(self):
        """Test is_async=False for sync functions."""
        def sync_func():
            pass

        func = Function(sync_func)
        assert func.is_async is False

    def test_async_function_is_async_true(self):
        """Test is_async=True for async functions."""
        async def async_func():
            pass

        func = Function(async_func)
        assert func.is_async is True

    def test_sync_function_signature_no_prefix(self):
        """Test sync function signature has no 'async' prefix."""
        def sync_func(a, b):
            return a + b

        func = Function(sync_func)
        assert func.signature.startswith("sync_func(")
        assert not func.signature.startswith("async ")

    def test_async_function_signature_has_prefix(self):
        """Test async function signature has 'async ' prefix."""
        async def async_func(a, b):
            return a + b

        func = Function(async_func)
        assert func.signature.startswith("async async_func(")


