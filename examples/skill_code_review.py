from cave_agent import CaveAgent
from cave_agent.skills import SkillDiscovery
from cave_agent.models import LiteLLMModel
from pathlib import Path
import os
import asyncio


# Sample code to review
SAMPLE_CODE = '''
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        return True
    return False
'''


async def main():
    # Initialize model
    model = LiteLLMModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        custom_llm_provider='openai'
    )

    # Get skills directory path
    skills_dir = Path(__file__).parent / "skills"
    skills = SkillDiscovery.from_directory(skills_dir)

    print("=== Agent Skills Example ===\n")

    # Create agent with skills from directory
    agent = CaveAgent(model=model, skills=skills)

    # Show loaded skills
    print("Loaded skills:")
    for skill in agent._skill_registry.list_skills():
        print(f"  - {skill.name}: {skill.description}")
    print()

    # Show activate_skill function in runtime
    print("Available functions:")
    print(agent.runtime.describe_functions())
    print()

    # Run the agent - it will use the skill when appropriate
    print("Asking agent to review code...\n")
    response = await agent.run(f"Review this code for security issues:\n```python\n{SAMPLE_CODE}\n```")

    print("Agent Response:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
