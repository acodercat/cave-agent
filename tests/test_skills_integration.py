"""
Integration tests for skills using agent.run().

Tests all skill features through natural language interactions:
- Skill activation
- Injection (functions, variables, types)
"""
import pytest
from pathlib import Path

from cave_agent import CaveAgent
from cave_agent.skills import SkillDiscovery
from cave_agent.runtime import PythonRuntime, Variable


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def skills_dir():
    """Path to the test skills directory."""
    return Path(__file__).parent / "skills"


@pytest.fixture
def data_analysis_skill(skills_dir):
    """Load the data-analysis skill."""
    return SkillDiscovery.from_file(skills_dir / "data-analysis" / "SKILL.md")


@pytest.fixture
def agent_with_skills(model, skills_dir):
    """Create agent with skills loaded and placeholder variables for retrieval."""
    # Pre-inject variables that tests will use for storing results
    runtime = PythonRuntime(
        variables=[
            Variable("instructions", description="Store skill instructions here"),
            Variable("stats", description="Store statistics results here"),
            Variable("outliers", description="Store outliers list here"),
            Variable("config", description="Store configuration here"),
            Variable("point", description="Store DataPoint instance here"),
            Variable("values", description="Store extracted values here"),
        ]
    )

    skills = SkillDiscovery.from_directory(skills_dir)
    return CaveAgent(
        model=model,
        skills=skills,
        runtime=runtime,
        max_steps=10
    )


# =============================================================================
# Skill Activation Tests
# =============================================================================

class TestSkillActivation:
    """Test skill activation through agent.run()."""

    @pytest.mark.asyncio
    async def test_activate_skill(self, agent_with_skills):
        """Test agent can activate a skill."""
        await agent_with_skills.run(
            "Activate the 'data-analysis' skill and store the instructions in a variable called 'instructions'."
        )

        instructions = agent_with_skills.runtime.retrieve('instructions')
        assert instructions is not None
        assert "Data Analysis Skill" in instructions
        assert "calculate_stats" in instructions


# =============================================================================
# Injected Function Tests
# =============================================================================

class TestInjectedFunctions:
    """Test using injected functions through agent.run()."""

    @pytest.mark.asyncio
    async def test_calculate_stats(self, agent_with_skills):
        """Test using calculate_stats function."""
        await agent_with_skills.run(
            "Activate 'data-analysis', then use calculate_stats on [10, 20, 30, 40, 50] "
            "and store the result in a variable called 'stats'."
        )

        stats = agent_with_skills.runtime.retrieve('stats')
        assert stats is not None
        assert stats['mean'] == 30.0
        assert stats['median'] == 30
        assert stats['min'] == 10
        assert stats['max'] == 50

    @pytest.mark.asyncio
    async def test_find_outliers(self, agent_with_skills):
        """Test using find_outliers function."""
        await agent_with_skills.run(
            "Activate 'data-analysis' and use find_outliers on [1, 2, 3, 4, 5, 100]. "
            "Store the result in a variable called 'outliers'."
        )

        outliers = agent_with_skills.runtime.retrieve('outliers')
        assert outliers is not None
        assert 100 in outliers


# =============================================================================
# Injected Variable Tests
# =============================================================================

class TestInjectedVariables:
    """Test using injected variables through agent.run()."""

    @pytest.mark.asyncio
    async def test_use_data_config(self, agent_with_skills):
        """Test accessing DATA_CONFIG variable."""
        await agent_with_skills.run(
            "Activate 'data-analysis' and copy the DATA_CONFIG variable to a new variable called 'config'."
        )

        config = agent_with_skills.runtime.retrieve('config')
        assert config is not None
        assert config['default_threshold'] == 1.5
        assert config['max_data_points'] == 10000
        assert 'csv' in config['supported_formats']


# =============================================================================
# Injected Type Tests
# =============================================================================

class TestInjectedTypes:
    """Test using injected types through agent.run()."""

    @pytest.mark.asyncio
    async def test_create_datapoint(self, agent_with_skills):
        """Test creating DataPoint instances."""
        await agent_with_skills.run(
            "Activate 'data-analysis' and create a DataPoint with value=42.5 "
            "and label='test'. Store it in a variable called 'point'."
        )

        point = agent_with_skills.runtime.retrieve('point')
        assert point is not None
        assert point.value == 42.5
        assert point.label == 'test'


# =============================================================================
# Multi-Turn Workflow Tests
# =============================================================================

class TestMultiTurnWorkflow:
    """Test multi-turn conversations with skills."""

    @pytest.mark.asyncio
    async def test_analysis_workflow(self, agent_with_skills):
        """Test a multi-turn analysis workflow."""
        # Turn 1: Activate skill
        await agent_with_skills.run("Activate the 'data-analysis' skill.")

        # Turn 2: Calculate stats on provided data
        await agent_with_skills.run(
            "Use calculate_stats on [10, 20, 30, 40, 50] and store the result in 'stats'."
        )

        stats = agent_with_skills.runtime.retrieve('stats')
        assert stats is not None
        assert 'mean' in stats
        assert stats['mean'] == 30.0

    @pytest.mark.asyncio
    async def test_full_workflow(self, agent_with_skills):
        """Test complete workflow using all features."""
        await agent_with_skills.run(
            "I need data analysis. Please:\n"
            "1. Activate 'data-analysis'\n"
            "2. Set values = [1, 2, 3, 4, 5, 100]\n"
            "3. Calculate statistics using calculate_stats and store in 'stats'\n"
            "4. Find any outliers using find_outliers and store in 'outliers'"
        )

        stats = agent_with_skills.runtime.retrieve('stats')
        outliers = agent_with_skills.runtime.retrieve('outliers')

        assert stats is not None
        assert 'mean' in stats
        assert outliers is not None
        # 100 should be detected as outlier
        assert 100 in outliers


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling through agent.run()."""

    @pytest.mark.asyncio
    async def test_nonexistent_skill(self, agent_with_skills):
        """Test handling of nonexistent skill."""
        response = await agent_with_skills.run(
            "Try to activate a skill called 'nonexistent-skill'."
        )

        assert response is not None
        content = response.content.lower()
        # Should mention error or not found
        assert any(word in content for word in ["error", "not found", "available", "doesn't exist"])
