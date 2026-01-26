import asyncio
import os
from pathlib import Path
from cave_agent import CaveAgent
from cave_agent.skills import SkillDiscovery
from cave_agent.models import LiteLLMModel


async def main():
    model = LiteLLMModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        custom_llm_provider="openai",
    )

    skills_dir = Path(__file__).parent / "skills"
    skills = SkillDiscovery.from_directory(skills_dir)

    print("=== Smart Home Assistant ===\n")

    agent = CaveAgent(model=model, skills=skills)

    print("Available skills:")
    for skill in agent._skill_registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")
    print()

    queries = [
        "What's the status of all the lights?",
        "Turn on the living room light to 70% brightness",
        "Is the front door locked?",
        "Lock the garage door and turn off the kitchen light",
    ]

    for query in queries:
        print(f"User: {query}")
        response = await agent.run(query)
        print(f"Assistant: {response.content}\n")

    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
