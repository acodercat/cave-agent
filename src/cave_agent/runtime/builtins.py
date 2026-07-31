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
    import sys

    # activate_skill runs inside the runtime namespace, but its own module
    # globals are not that namespace. The runtime injects ``_skill_store`` into
    # the cell's namespace, so walk the call stack from the immediate caller and
    # take the first frame whose globals carry it. Walking (rather than a fixed
    # _getframe(1)) also finds the store when activation is routed through a
    # helper imported from another module, and needs no process-global
    # get_ipython() — so the in-process InteractiveShell and the IPyKernel
    # subprocess behave identically. If no frame carries the store (activation
    # from a context detached from any cell), ``ns`` stays empty and the lookup
    # below raises a clean "not found" KeyError.
    ns: dict = {}
    frame = sys._getframe(1)
    while frame is not None:
        if "_skill_store" in frame.f_globals:
            ns = frame.f_globals
            break
        frame = frame.f_back

    store = ns.get("_skill_store", {})
    if skill_name not in store:
        available = list(store.keys())
        raise KeyError(f"Skill '{skill_name}' not found. Available skills: {available}")

    return store[skill_name]
