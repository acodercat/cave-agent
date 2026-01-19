"""
Integration tests for skills using agent.run().

Tests all skill features through natural language interactions:
- Skill activation
- Injection (functions, variables, types)
- Script execution (run_skill_script)
- Reference reading (read_skill_reference)
- Asset reading (read_skill_asset)
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
            Variable("analysis", description="Store analysis results here"),
            Variable("validation", description="Store validation results here"),
            Variable("normalized", description="Store normalized data here"),
            Variable("guide", description="Store guide content here"),
            Variable("api_doc", description="Store API documentation here"),
            Variable("csv_data", description="Store CSV data here"),
            Variable("values", description="Store extracted values here"),
        ]
    )

    return CaveAgent(
        model=model,
        skills_dir=skills_dir,
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
# Script Execution Tests
# =============================================================================

class TestScriptExecution:
    """Test running scripts through agent.run()."""

    @pytest.mark.asyncio
    async def test_run_analyze_script(self, agent_with_skills):
        """Test running analyze.py script."""
        await agent_with_skills.run(
            "Use run_skill_script to run 'analyze.py' from 'data-analysis' "
            "with data=[10, 20, 30, 40, 50, 100]. Store the result in 'analysis'."
        )

        analysis = agent_with_skills.runtime.retrieve('analysis')
        assert analysis is not None
        assert 'stats' in analysis
        assert 'outliers' in analysis
        assert analysis['stats']['count'] == 6

    @pytest.mark.asyncio
    async def test_run_validate_script(self, agent_with_skills):
        """Test running validate.py script."""
        await agent_with_skills.run(
            "Run the validate.py script from 'data-analysis' with "
            "data=[10, 20, 30], min_value=0, max_value=100. Store result in 'validation'."
        )

        validation = agent_with_skills.runtime.retrieve('validation')
        assert validation is not None
        assert validation['is_valid'] is True
        assert validation['errors'] == []

    @pytest.mark.asyncio
    async def test_run_transform_script(self, agent_with_skills):
        """Test running transform.py script."""
        await agent_with_skills.run(
            "Run transform.py from 'data-analysis' with data=[0, 50, 100] "
            "and operation='normalize'. Store the result in 'normalized'."
        )

        normalized = agent_with_skills.runtime.retrieve('normalized')
        assert normalized is not None
        assert normalized == [0.0, 0.5, 1.0]


# =============================================================================
# Reference Reading Tests
# =============================================================================

class TestReferenceReading:
    """Test reading references through agent.run()."""

    @pytest.mark.asyncio
    async def test_read_guide_reference(self, agent_with_skills):
        """Test reading GUIDE.md reference."""
        await agent_with_skills.run(
            "Use read_skill_reference to read 'GUIDE.md' from 'data-analysis'. "
            "Store the content in a variable called 'guide'."
        )

        guide = agent_with_skills.runtime.retrieve('guide')
        assert guide is not None
        assert "Data Analysis Skill Guide" in guide
        assert "Best Practices" in guide

    @pytest.mark.asyncio
    async def test_read_api_reference(self, agent_with_skills):
        """Test reading API.md reference."""
        await agent_with_skills.run(
            "Read the API.md reference from 'data-analysis'. "
            "Store the content in a variable called 'api_doc'."
        )

        api_doc = agent_with_skills.runtime.retrieve('api_doc')
        assert api_doc is not None
        assert "API Reference" in api_doc
        assert "calculate_stats" in api_doc


# =============================================================================
# Asset Reading Tests
# =============================================================================

class TestAssetReading:
    """Test reading assets through agent.run()."""

    @pytest.mark.asyncio
    async def test_read_config_json(self, agent_with_skills):
        """Test reading config.json asset."""
        await agent_with_skills.run(
            "Use read_skill_asset to read 'config.json' from 'data-analysis'. "
            "Parse the JSON and store it in a variable called 'config'."
        )

        config = agent_with_skills.runtime.retrieve('config')
        assert config is not None
        assert config['version'] == '1.0.0'
        assert config['settings']['default_threshold'] == 1.5

    @pytest.mark.asyncio
    async def test_read_csv_asset(self, agent_with_skills):
        """Test reading sample-data.csv asset."""
        await agent_with_skills.run(
            "Read the 'sample-data.csv' asset from 'data-analysis'. "
            "Store the raw content in a variable called 'csv_data'."
        )

        csv_data = agent_with_skills.runtime.retrieve('csv_data')
        assert csv_data is not None
        # Should be bytes
        content = csv_data.decode('utf-8') if isinstance(csv_data, bytes) else csv_data
        assert "id,value,label,timestamp" in content
        assert "outlier" in content


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

        # Turn 2: Load data and extract values
        await agent_with_skills.run(
            "Read the sample-data.csv asset and extract the 'value' column into a list called 'values'."
        )

        values = agent_with_skills.runtime.retrieve('values')
        assert values is not None
        assert len(values) == 7

        # Turn 3: Analyze
        await agent_with_skills.run(
            "Use calculate_stats on the 'values' list and store the result in 'stats'."
        )

        stats = agent_with_skills.runtime.retrieve('stats')
        assert stats is not None
        assert 'mean' in stats

    @pytest.mark.asyncio
    async def test_full_workflow(self, agent_with_skills):
        """Test complete workflow using all features."""
        await agent_with_skills.run(
            "I need a complete data analysis. Please:\n"
            "1. Activate 'data-analysis'\n"
            "2. Read the sample-data.csv asset\n"
            "3. Extract the numeric 'value' column into a list called 'values'\n"
            "4. Calculate statistics using calculate_stats and store in 'stats'\n"
            "5. Find any outliers using find_outliers and store in 'outliers'"
        )

        stats = agent_with_skills.runtime.retrieve('stats')
        outliers = agent_with_skills.runtime.retrieve('outliers')

        assert stats is not None
        assert 'mean' in stats
        assert outliers is not None
        # 100.0 should be detected as outlier
        assert 100.0 in outliers


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

    @pytest.mark.asyncio
    async def test_nonexistent_script(self, agent_with_skills):
        """Test handling of nonexistent script."""
        response = await agent_with_skills.run(
            "Run a script called 'nonexistent.py' from 'data-analysis'."
        )

        assert response is not None
        content = response.content.lower()
        # Should mention error, not found, or indicate inability to run the script
        assert any(word in content for word in [
            "error", "not found", "doesn't exist", "failed", "cannot", "couldn't",
            "unable", "no such", "missing", "unavailable", "not available", "exception"
        ])
