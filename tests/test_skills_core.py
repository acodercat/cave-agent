import pytest
import tempfile
from pathlib import Path

from cave_agent.skills import Skill, SkillRegistry, SkillDiscovery
from cave_agent import CaveAgent
from cave_agent.runtime import IPythonRuntime, IPyKernelRuntime, Function, Variable, Type
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
# Skill Tests
# =============================================================================

class TestSkill:
    def test_skill_creation(self):
        """Test creating a Skill directly."""
        skill = Skill(
            name="my-skill",
            description="A custom skill",
            body_content="# Instructions\nDo something."
        )
        assert skill.name == "my-skill"
        assert skill.description == "A custom skill"
        assert skill.body_content == "# Instructions\nDo something."
        assert skill.functions == []
        assert skill.variables == []
        assert skill.types == []

    def test_skill_with_functions(self):
        """Test Skill with injected functions."""
        def helper(x):
            return x * 2

        skill = Skill(
            name="func-skill",
            description="Skill with functions",
            functions=[Function(helper)]
        )
        assert len(skill.functions) == 1
        assert skill.functions[0].name == "helper"

    def test_skill_with_variables(self):
        """Test Skill with injected variables."""
        skill = Skill(
            name="var-skill",
            description="Skill with variables",
            variables=[Variable("config", value={"key": "value"})]
        )
        assert len(skill.variables) == 1
        assert skill.variables[0].name == "config"

    def test_skill_with_types(self):
        """Test Skill with injected types."""
        class MyClass:
            pass

        skill = Skill(
            name="type-skill",
            description="Skill with types",
            types=[Type(MyClass)]
        )
        assert len(skill.types) == 1
        assert skill.types[0].name == "MyClass"

    def test_skill_with_all_injections(self):
        """Test Skill with functions, variables, and types."""
        def func():
            pass

        class MyType:
            pass

        skill = Skill(
            name="full-skill",
            description="Skill with everything",
            body_content="Instructions",
            functions=[Function(func)],
            variables=[Variable("var", value=42)],
            types=[Type(MyType)]
        )
        assert skill.name == "full-skill"
        assert len(skill.functions) == 1
        assert len(skill.variables) == 1
        assert len(skill.types) == 1


# =============================================================================
# SkillDiscovery Tests
# =============================================================================

class TestSkillDiscoveryFromFile:
    def test_load_valid_skill(self, skill_file):
        """Test loading a valid skill file."""
        skill = SkillDiscovery.from_file(skill_file)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert "Test Skill Instructions" in skill.body_content

    def test_load_skill_with_injection(self, temp_dir):
        """Test loading skill with injection.py."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
from cave_agent.runtime import Function, Variable, Type

def helper(x):
    return x * 2

CONFIG = {"key": "value"}

class Result:
    pass

__exports__ = [
    Function(helper),
    Variable("CONFIG", value=CONFIG),
    Type(Result),
]
""")

        skill = SkillDiscovery.from_file(temp_dir / "SKILL.md")

        assert skill.name == "test"
        assert len(skill.functions) == 1
        assert len(skill.variables) == 1
        assert len(skill.types) == 1
        assert skill.functions[0].name == "helper"
        assert skill.variables[0].name == "CONFIG"
        assert skill.types[0].name == "Result"

    def test_load_skill_without_injection(self, skill_file):
        """Test loading skill without injection.py."""
        skill = SkillDiscovery.from_file(skill_file)

        assert skill.functions == []
        assert skill.variables == []
        assert skill.types == []

    def test_missing_name_raises_error(self, temp_dir):
        """Test that missing name raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\ndescription: No name\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'name'" in str(exc_info.value)

    def test_missing_description_raises_error(self, temp_dir):
        """Test that missing description raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: no-desc\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Missing required field 'description'" in str(exc_info.value)

    def test_no_frontmatter_raises_error(self, temp_dir):
        """Test that missing frontmatter raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("Just content, no frontmatter")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "No YAML frontmatter found" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, temp_dir):
        """Test that invalid YAML raises SkillDiscovery.Error."""
        path = temp_dir / "SKILL.md"
        path.write_text("---\nname: [invalid yaml\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "Invalid YAML" in str(exc_info.value)

    def test_nonexistent_file_raises_error(self, temp_dir):
        """Test that nonexistent file raises SkillDiscovery.Error."""
        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(temp_dir / "nonexistent.md")
        assert "Skill file not found" in str(exc_info.value)

    def test_missing_exports_raises_error(self, temp_dir):
        """Test missing __exports__ raises SkillDiscovery.Error."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
def helper():
    pass
# No __exports__ defined
""")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(temp_dir / "SKILL.md")
        assert "Missing __exports__" in str(exc_info.value)

    def test_invalid_injection_module_raises_error(self, temp_dir):
        """Test invalid injection module raises SkillDiscovery.Error."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test\n---\nContent"
        )
        (temp_dir / "injection.py").write_text("""
# Invalid Python syntax
def broken(
""")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(temp_dir / "SKILL.md")
        assert "Error loading injection module" in str(exc_info.value)


class TestSkillDiscoveryValidation:
    def test_name_too_long_rejected(self, temp_dir):
        """Test that name exceeding 64 chars raises error."""
        path = temp_dir / "SKILL.md"
        long_name = "a" * 65
        path.write_text(f"---\nname: {long_name}\ndescription: Desc\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "exceeds 64 characters" in str(exc_info.value)

    def test_description_too_long_rejected(self, temp_dir):
        """Test that description exceeding 1024 chars raises error."""
        path = temp_dir / "SKILL.md"
        long_desc = "a" * 1025
        path.write_text(f"---\nname: test\ndescription: {long_desc}\n---\nContent")

        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_file(path)
        assert "exceeds 1024 characters" in str(exc_info.value)


class TestSkillDiscoveryFromDirectory:
    def test_from_directory(self, temp_dir):
        """Test discovering skills from directory."""
        (temp_dir / "SKILL.md").write_text(
            "---\nname: root-skill\ndescription: Root skill\n---\nRoot content"
        )

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
        """Test discovering from nonexistent directory raises error."""
        with pytest.raises(SkillDiscovery.Error) as exc_info:
            SkillDiscovery.from_directory(Path("/nonexistent/path"))
        assert "Skills directory not found" in str(exc_info.value)


# =============================================================================
# SkillRegistry Tests
# =============================================================================

class TestSkillRegistry:
    def test_add_and_get_skill(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test skill")
        registry.add_skill(skill)

        assert registry.get_skill("test") is skill
        assert registry.get_skill("nonexistent") is None

    def test_add_skills(self):
        registry = SkillRegistry()
        skills = [
            Skill(name="skill1", description="First"),
            Skill(name="skill2", description="Second"),
        ]
        registry.add_skills(skills)
        assert len(registry.list_skills()) == 2

    def test_describe_skills(self):
        registry = SkillRegistry()
        registry.add_skill(Skill(name="my-skill", description="My description"))

        description = registry.describe_skills()
        assert "my-skill" in description
        assert "My description" in description

    def test_describe_skills_empty(self):
        registry = SkillRegistry()
        assert registry.describe_skills() == "No skills available"


class TestBuildSkillStore:
    """Test SkillRegistry.build_skill_store()."""

    def test_empty_registry(self):
        registry = SkillRegistry()
        assert registry.build_skill_store() == {}

    def test_store_contains_body_content(self):
        registry = SkillRegistry()
        registry.add_skill(Skill(name="test", description="Test", body_content="Instructions"))
        store = registry.build_skill_store()

        assert "test" in store
        assert store["test"]["body_content"] == "Instructions"

    def test_store_contains_function_exports(self):
        def helper(x):
            return x * 2

        registry = SkillRegistry()
        registry.add_skill(Skill(name="test", description="Test", functions=[Function(helper)]))
        store = registry.build_skill_store()

        assert "helper" in store["test"]["exports"]
        assert store["test"]["exports"]["helper"](5) == 10

    def test_store_contains_variable_exports(self):
        registry = SkillRegistry()
        registry.add_skill(Skill(
            name="test", description="Test",
            variables=[Variable("CONFIG", value={"key": "val"})],
        ))
        store = registry.build_skill_store()

        assert store["test"]["exports"]["CONFIG"] == {"key": "val"}

    def test_store_contains_type_exports(self):
        class MyType:
            pass

        registry = SkillRegistry()
        registry.add_skill(Skill(name="test", description="Test", types=[Type(MyType)]))
        store = registry.build_skill_store()

        assert store["test"]["exports"]["MyType"] is MyType

    def test_store_multiple_skills(self):
        registry = SkillRegistry()
        registry.add_skill(Skill(name="a", description="A", body_content="A instructions"))
        registry.add_skill(Skill(name="b", description="B", body_content="B instructions"))
        store = registry.build_skill_store()

        assert len(store) == 2
        assert store["a"]["body_content"] == "A instructions"
        assert store["b"]["body_content"] == "B instructions"


# =============================================================================
# CaveAgent Skills Integration Tests
# =============================================================================

class TestAgentSkillsInit:
    def test_agent_with_skills_list(self, mock_model):
        skill = Skill(name="test-skill", description="Test")
        agent = CaveAgent(model=mock_model, skills=[skill])
        assert agent._skill_registry.get_skill("test-skill") is not None

    def test_agent_with_skills_from_discovery(self, temp_dir, mock_model):
        (temp_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Test skill\n---\nContent"
        )
        skills = SkillDiscovery.from_directory(temp_dir)
        agent = CaveAgent(model=mock_model, skills=skills)
        assert agent._skill_registry.get_skill("test") is not None

    def test_agent_no_skills(self, mock_model):
        agent = CaveAgent(model=mock_model)
        assert agent._skill_registry.list_skills() == []


class TestAgentSkillsSystemPrompt:
    def test_skills_in_system_prompt(self, mock_model):
        skill = Skill(name="my-skill", description="My description")
        agent = CaveAgent(model=mock_model, skills=[skill])
        prompt = agent.build_system_prompt()

        assert "my-skill" in prompt
        assert "My description" in prompt

    def test_no_skills_in_system_prompt(self, mock_model):
        agent = CaveAgent(model=mock_model)
        prompt = agent.build_system_prompt()
        assert "No skills available" in prompt


class TestAgentSkillsRuntime:
    def test_activate_skill_function_injected(self, mock_model):
        """activate_skill function is injected into runtime."""
        skill = Skill(name="test", description="Test")
        agent = CaveAgent(model=mock_model, skills=[skill])
        assert "activate_skill" in agent.runtime.describe_functions()

    def test_activate_skill_not_injected_without_skills(self, mock_model):
        agent = CaveAgent(model=mock_model)
        assert "activate_skill" not in agent.runtime.describe_functions()

    @pytest.mark.asyncio
    async def test_activate_skill_execution(self, mock_model):
        """activate_skill works in IPythonRuntime."""
        skill = Skill(
            name="my-skill",
            description="Test",
            body_content="Skill instructions here",
        )
        agent = CaveAgent(model=mock_model, skills=[skill])

        result = await agent.runtime.execute('output = activate_skill("my-skill")')
        assert result.success

        result = await agent.runtime.execute("print(output)")
        assert "Skill instructions here" in result.stdout

    @pytest.mark.asyncio
    async def test_activate_skill_injects_exports(self, mock_model):
        """activate_skill injects functions/variables into namespace."""
        def helper(x):
            return x * 2

        skill = Skill(
            name="my-skill",
            description="Test",
            body_content="Instructions",
            functions=[Function(helper)],
            variables=[Variable("data", value=[1, 2, 3])],
        )
        agent = CaveAgent(model=mock_model, skills=[skill])

        await agent.runtime.execute('activate_skill("my-skill")')

        result = await agent.runtime.execute("print(helper(5))")
        assert "10" in result.stdout

        result = await agent.runtime.execute("print(sum(data))")
        assert "6" in result.stdout

    @pytest.mark.asyncio
    async def test_activate_skill_not_found(self, mock_model):
        """activate_skill raises KeyError for unknown skill."""
        skill = Skill(name="test", description="Test")
        agent = CaveAgent(model=mock_model, skills=[skill])

        result = await agent.runtime.execute('activate_skill("nonexistent")')
        assert not result.success
        assert "nonexistent" in result.stdout


# =============================================================================
# IPyKernelRuntime Skills Tests
# =============================================================================

class TestAgentSkillsIPyKernel:
    """Test CaveAgent skills with IPyKernelRuntime."""

    def test_activate_skill_function_injected(self, mock_model):
        skill = Skill(name="test", description="Test")
        runtime = IPyKernelRuntime()
        agent = CaveAgent(model=mock_model, runtime=runtime, skills=[skill])
        assert "activate_skill" in agent.runtime.describe_functions()

    @pytest.mark.asyncio
    async def test_activate_skill_execution_in_kernel(self, mock_model):
        """activate_skill works in IPyKernelRuntime."""
        skill = Skill(
            name="my-skill",
            description="Test",
            body_content="Skill instructions here",
        )
        runtime = IPyKernelRuntime()
        agent = CaveAgent(model=mock_model, runtime=runtime, skills=[skill])

        await runtime.start()
        try:
            result = await agent.runtime.execute('output = activate_skill("my-skill")')
            assert result.success

            result = await agent.runtime.execute("print(output)")
            assert "Skill instructions here" in result.stdout
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_activate_skill_injects_exports_in_kernel(self, mock_model):
        """activate_skill injects functions/variables into kernel namespace."""
        def helper(x):
            return x * 2

        skill = Skill(
            name="my-skill",
            description="Test",
            body_content="Instructions",
            functions=[Function(helper)],
            variables=[Variable("data", value=[1, 2, 3])],
        )
        runtime = IPyKernelRuntime()
        agent = CaveAgent(model=mock_model, runtime=runtime, skills=[skill])

        await runtime.start()
        try:
            await agent.runtime.execute('activate_skill("my-skill")')

            result = await agent.runtime.execute("print(helper(5))")
            assert result.success
            assert "10" in result.stdout

            result = await agent.runtime.execute("print(sum(data))")
            assert result.success
            assert "6" in result.stdout
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_activate_nonexistent_skill_in_kernel(self, mock_model):
        """activate_skill raises KeyError for unknown skill in kernel."""
        skill = Skill(name="real-skill", description="Test")
        runtime = IPyKernelRuntime()
        agent = CaveAgent(model=mock_model, runtime=runtime, skills=[skill])

        await runtime.start()
        try:
            result = await agent.runtime.execute('activate_skill("nonexistent")')
            assert not result.success
            assert "nonexistent" in result.stdout
        finally:
            await runtime.stop()


# =============================================================================
# Function.is_async Tests
# =============================================================================

class TestFunctionIsAsync:
    def test_sync_function_is_async_false(self):
        def sync_func():
            pass

        func = Function(sync_func)
        assert func.is_async is False

    def test_async_function_is_async_true(self):
        async def async_func():
            pass

        func = Function(async_func)
        assert func.is_async is True
