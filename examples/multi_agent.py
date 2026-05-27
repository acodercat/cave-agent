"""Multi-Agent Data Processing Pipeline (Supervisor Pattern).

Demonstrates orchestrator + sub-agent coordination with Rich display.

Usage:
    cd cave-agent && uv run python examples/multi_agent.py
"""

import asyncio
import os

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

from cave_agent import CaveAgent
from cave_agent.models.openai import OpenAIServerModel
from cave_agent.runtime import IPythonRuntime, Variable

console = Console()

model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


async def main():
    raw_data = [
        {"name": "Alice", "age": 30, "dept": "Engineering", "salary": 50000, "rating": 4.2},
        {"name": "Bob", "age": None, "dept": "Marketing", "salary": 60000, "rating": 3.8},
        {"name": "Charlie", "age": 25, "dept": "Engineering", "salary": None, "rating": 4.5},
        {"name": "Diana", "age": 35, "dept": "Sales", "salary": 70000, "rating": None},
        {"name": "Eve", "age": None, "dept": None, "salary": None, "rating": 3.1},
        {"name": "Frank", "age": 28, "dept": "Engineering", "salary": 55000, "rating": 4.0},
        {"name": "Grace", "age": 32, "dept": "Marketing", "salary": None, "rating": 4.7},
        {"name": "Henry", "age": None, "dept": "Sales", "salary": 48000, "rating": 3.5},
        {"name": "Ivy", "age": 27, "dept": "Engineering", "salary": 62000, "rating": 4.3},
    ]

    # -- Sub-Agents --------------------------------------------------------

    cleaner = CaveAgent(
        model,
        runtime=IPythonRuntime(
            variables=[
                Variable("data", [], "Input: list of dicts with keys 'name', 'age', 'dept', 'salary', 'rating'"),
                Variable("cleaned_data", [], "Output: list of dicts with no None values"),
            ]
        ),
        display=False,
    )

    analyzer = CaveAgent(
        model,
        runtime=IPythonRuntime(
            variables=[
                Variable("data", [], "Input: list of dicts with keys 'name', 'age', 'dept', 'salary', 'rating'"),
                Variable("insights", {}, "Output: dict with statistics"),
            ]
        ),
        display=False,
    )

    # -- Orchestrator ------------------------------------------------------

    orchestrator = CaveAgent(
        model,
        runtime=IPythonRuntime(
            variables=[
                Variable("raw_data", raw_data, "Raw dataset with potential null values"),
                Variable("cleaner", cleaner, "Cleaner agent: call cleaner.runtime.update_variable('data', value) to set input, await cleaner.run('instruction') to execute, await cleaner.runtime.retrieve('cleaned_data') to get output"),
                Variable("analyzer", analyzer, "Analyzer agent: call analyzer.runtime.update_variable('data', value) to set input, await analyzer.run('instruction') to execute, await analyzer.runtime.retrieve('insights') to get output"),
                Variable("cleaned_data", [], "Cleaned data from cleaner agent"),
                Variable("insights", {}, "Insights from analyzer agent"),
            ]
        ),
        instructions="You are a supervisor agent. You coordinate the work of the cleaner and analyzer agents.",
        max_steps=20,
        max_exec_output=50000,
    )

    # -- UI: Header --------------------------------------------------------

    console.print()
    console.print(Rule("[bold cyan]Multi-Agent Pipeline[/]"))
    console.print()

    # Agent architecture (left) + Raw data (right)
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.syntax import Syntax

    code = '''\
orchestrator = CaveAgent(
  model,
  runtime=IPythonRuntime(variables=[
    Variable("raw_data",  raw_data),
    Variable("cleaner",   cleaner_agent),
    Variable("analyzer",  analyzer_agent),
    Variable("cleaned_data", []),
    Variable("insights",     {}),
  ]),
  instructions="Coordinate cleaner → analyzer",
)'''
    panel_height = max(len(code.splitlines()) + 2, len(raw_data) + 3)

    code_panel = Panel(
        Syntax(code, "python", theme="native", line_numbers=False, background_color="default"),
        title="[bold]Agent Setup[/]",
        border_style="dim",
        expand=True,
        height=panel_height,
    )

    data_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    data_table.add_column("Name")
    data_table.add_column("Age", justify="right")
    data_table.add_column("Dept")
    data_table.add_column("Salary", justify="right")
    data_table.add_column("Rating", justify="right")
    for record in raw_data:
        def fmt(val, is_number=False):
            if val is None:
                return Text("None", style="red")
            if is_number and isinstance(val, (int, float)):
                return f"{val:,}" if isinstance(val, int) else f"{val:.1f}"
            return str(val)
        data_table.add_row(
            record["name"],
            fmt(record["age"]),
            fmt(record["dept"]),
            fmt(record["salary"], is_number=True),
            fmt(record["rating"]),
        )
    data_panel = Panel(data_table, title="[bold]Raw Data[/]", border_style="dim", expand=True, height=panel_height)

    from rich.table import Table as LayoutTable
    layout = LayoutTable.grid(expand=True)
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(code_panel, data_panel)
    console.print(layout)

    console.print()
    console.print(Rule("[bold cyan]Running Pipeline[/]"))

    # -- Execute -----------------------------------------------------------

    await orchestrator.run("Clean the raw_data using cleaner, then analyze it using analyzer")

    # -- UI: Results -------------------------------------------------------

    final_cleaned = await orchestrator.runtime.retrieve("cleaned_data")
    final_insights = await orchestrator.runtime.retrieve("insights")

    console.print()
    console.print(Rule("[bold green]Results[/]"))
    console.print()

    # Cleaned data
    result_table = Table(title="[dim]Cleaned Data[/]", show_header=True, header_style="bold", box=None, padding=(0, 1))
    result_table.add_column("Name")
    result_table.add_column("Age", justify="right")
    result_table.add_column("Dept")
    result_table.add_column("Salary", justify="right")
    result_table.add_column("Rating", justify="right")
    for record in final_cleaned:
        result_table.add_row(
            record.get("name", ""),
            str(record.get("age", "")),
            str(record.get("dept", "")),
            f"{record['salary']:,}" if record.get("salary") else "",
            f"{record['rating']:.1f}" if record.get("rating") else "",
        )
    console.print(result_table)
    console.print()

    # Insights
    if final_insights:
        insight_table = Table(title="[dim]Insights[/]", show_header=True, header_style="bold", border_style="green", padding=(0, 1))
        insight_table.add_column("Metric", style="bold")
        insight_table.add_column("Value", justify="right")
        for key, value in final_insights.items():
            display_value = f"{value:,.1f}" if isinstance(value, float) else str(value)
            insight_table.add_row(key.replace("_", " ").title(), display_value)
        console.print(insight_table)
        console.print()

    # Summary
    removed = len(raw_data) - len(final_cleaned)
    console.print(Text.assemble(
        ("  ", ""),
        (f"{len(raw_data)}", "bold"),
        (" raw → ", "dim"),
        (f"{len(final_cleaned)}", "bold green"),
        (" cleaned", "dim"),
        (f"  ({removed} removed)", "dim red"),
    ))
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
