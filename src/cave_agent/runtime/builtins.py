"""Built-in functions injected into the runtime namespace.

These functions are called by LLM-generated code inside the execution
environment (IPython shell or IPyKernel process), not by host-side code.
"""



def activate_skill(skill_name: str) -> str:
    """Activate a skill and return its instructions.

    Call this function ONCE when you need specialized guidance for a task.
    Print the returned value to see the skill's instructions, then follow
    them to complete the task. Do NOT call again for the same skill.

    Args:
        skill_name: The exact name of the skill to activate (from the skills list)

    Returns:
        The skill's instructions and guidance

    Raises:
        KeyError: If skill is not found
    """

    from IPython import get_ipython
    ns = get_ipython().user_ns
    store = ns.get("_skill_store", {})
    skill = store.get(skill_name)
    if not skill:
        available = list(store.keys())
        raise KeyError(f"Skill '{skill_name}' not found. Available skills: {available}")

    ns.update(skill["exports"])
    return skill["body_content"]
