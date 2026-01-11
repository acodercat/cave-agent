import asyncio
import os
from cave_agent import CaveAgent
from cave_agent.models import LiteLLMModel
from cave_agent.runtime import PythonRuntime, Variable


# Initialize LLM model
model = LiteLLMModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    custom_llm_provider="openai"
)


async def main():
    # ==========================================================================
    # Raw data with issues (nulls, incomplete records)
    # ==========================================================================
    raw_data = [
        {"name": "Alice", "age": 30, "salary": 50000},
        {"name": "Bob", "age": None, "salary": 60000},
        {"name": "Charlie", "age": 25, "salary": None},
        {"name": "Diana", "age": 35, "salary": 70000},
        {"name": "Eve", "age": None, "salary": None},
        {"name": "Frank", "age": 28, "salary": 55000},
    ]

    print("=" * 60)
    print("Multi-Agent Data Processing Pipeline (Supervisor Pattern)")
    print("=" * 60)
    print(f"\nRaw data ({len(raw_data)} records):")
    for record in raw_data:
        print(f"  {record}")

    # ==========================================================================
    # Create Sub-Agents
    # ==========================================================================

    # Cleaner Agent: Removes records with null values
    cleaner = CaveAgent(
        model,
        runtime=PythonRuntime(
            variables=[
                Variable("data", [], "Input: list of dicts with keys 'name', 'age', 'salary'"),
                Variable("cleaned_data", [], "Output: list of dicts with no None values in any field"),
            ]
        ),
    )

    # Analyzer Agent: Computes statistics
    analyzer = CaveAgent(
        model,
        runtime=PythonRuntime(
            variables=[
                Variable("data", [], "Input: list of dicts with keys 'name', 'age', 'salary'"),
                Variable("insights", {}, "Output: dict with computed statistics like total_records, avg_age, avg_salary"),
            ]
        ),
    )

    # ==========================================================================
    # Create Orchestrator Agent with sub-agents as variables
    # ==========================================================================


    orchestrator_agent_identity = """
    You are a supervisor agent. You coordinate the work of the cleaner and analyzer agents. You solve tasks by writing and executing Python code using the provided functions, variables, and their methods.
    """
    orchestrator = CaveAgent(
        model,
        runtime=PythonRuntime(
            variables=[
                # Input data
                Variable("raw_data", raw_data, "Raw dataset with potential null values"),
                # Sub-agents (injected as first-class objects)
                Variable("cleaner", cleaner, "Cleaner agent: call cleaner.runtime.update_variable('data', value) to set input, await cleaner.run('instruction') to execute with a task instruction string, cleaner.runtime.retrieve('cleaned_data') to get output"),
                Variable("analyzer", analyzer, "Analyzer agent: call analyzer.runtime.update_variable('data', value) to set input, await analyzer.run('instruction') to execute with a task instruction string, analyzer.runtime.retrieve('insights') to get output"),
                # Results
                Variable("cleaned_data", [], "Cleaned data from cleaner agent"),
                Variable("insights", {}, "Insights from analyzer agent"),
            ]
        ),
        agent_identity=orchestrator_agent_identity,
        max_steps=20,
        max_execution_result_length=50000
    )

    # ==========================================================================
    # Run Orchestrator - it will control sub-agents via code
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Orchestrator controlling sub-agents...")
    print("-" * 60)

    await orchestrator.run("Clean the raw_data using cleaner, then analyze it using analyzer")

    # ==========================================================================
    # Retrieve final results from orchestrator
    # ==========================================================================
    final_cleaned = orchestrator.runtime.retrieve("cleaned_data")
    final_insights = orchestrator.runtime.retrieve("insights")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"\nCleaned data ({len(final_cleaned)} records):")
    for record in final_cleaned:
        print(f"  {record}")

    print(f"\nInsights:")
    for key, value in final_insights.items():
        print(f"  {key}: {value}")

    print(f"\nSummary:")
    print(f"  Raw records:     {len(raw_data)}")
    print(f"  Cleaned records: {len(final_cleaned)}")
    print(f"  Records removed: {len(raw_data) - len(final_cleaned)}")


if __name__ == "__main__":
    asyncio.run(main())
