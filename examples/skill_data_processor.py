import asyncio
import os
from pathlib import Path

from cave_agent import CaveAgent
from cave_agent.skills import SkillDiscovery
from cave_agent.models import LiteLLMModel


async def main():
    # Initialize model
    model = LiteLLMModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        custom_llm_provider="openai",
    )

    # Get skills directory path
    skills_dir = Path(__file__).parent / "skills"
    skills = SkillDiscovery.from_directory(skills_dir)

    print("=== Sales Analysis Assistant ===\n")

    # Create agent with skills
    agent = CaveAgent(model=model, skills=skills)

    # Show available skills
    print("Available skills:")
    for skill in agent._skill_registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")
    print()

    # Natural conversation with the agent
    queries = [
        # Query 1: Overall sales analysis
        "How are our overall sales doing?",

        # Query 2: Regional breakdown
        "How is the north region performing against target?",

        # Query 3: Commission calculation
        "Calculate the commissions for electronics sales.",
    ]

    for query in queries:
        print(f"User: {query}")
        response = await agent.run(query)
        print(f"Assistant: {response.content}\n")

    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
