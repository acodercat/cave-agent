import pytest
from cave_agent import CaveAgent
from cave_agent.python_runtime import PythonRuntime, Variable

class DataAnalyzer:
    """A data analyzer that provides statistical analysis for numerical data."""
    
    def analyze(self, data: list) -> dict:
        """Calculate basic statistical measures for a list of numbers."""
        return {
            'min': min(data),
            'max': max(data),
            'avg': sum(data) / len(data),
            'len': len(data)
        }

@pytest.fixture
def analyzer():
    return DataAnalyzer()

@pytest.fixture
def numbers():
    return [3, 1, 4, 1, 5, 9, 2, 6, 5]

@pytest.fixture
def multi_turn_agent(model, analyzer, numbers):
    # Create Variable objects
    analyzer_var = Variable(
        name="analyzer",
        value=analyzer,
        description="Tool for analyzing numerical data\nusage: stats = analyzer.analyze(numbers)"
    )
    
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input data to analyze\nusage: print(numbers)"
    )
    
    stats_var = Variable(
        name="stats",
        description="Store analysis results here\nusage: stats = analyzer.analyze(numbers)"
    )
    
    # Create runtime
    runtime = PythonRuntime(
        variables=[analyzer_var, numbers_var, stats_var]
    )
    
    return CaveAgent(
        model,
        runtime=runtime
    )

@pytest.mark.asyncio
async def test_basic_analysis(multi_turn_agent):
    await multi_turn_agent.run("Analyze the numbers and store results in 'stats'")
    stats = multi_turn_agent.runtime.get_variable_value('stats')
    
    assert stats['min'] == 1
    assert stats['max'] == 9
    assert stats['avg'] == 4.0
    assert stats['len'] == 9

@pytest.mark.asyncio
async def test_multi_turn_conversation(multi_turn_agent):
    # First turn
    await multi_turn_agent.run("Analyze the numbers and store results in 'stats'")
    stats = multi_turn_agent.runtime.get_variable_value('stats')
    assert stats is not None
    assert all(k in stats for k in ['min', 'max', 'avg', 'len'])
    
    # Second turn
    response = await multi_turn_agent.run("What is the average value in the stats?")
    assert response is not None
    assert "4" in response.content.lower() or "four" in response.content.lower()
    
    # Third turn
    response = await multi_turn_agent.run("Is the maximum value (9) significantly higher than the average?")
    assert response is not None
    assert any(word in response.content.lower() for word in ['yes', 'higher', 'greater', 'more'])