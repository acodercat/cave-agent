from pathlib import Path
from typing import Any
import inspect

from ..runtime import PythonRuntime
from ..runtime.executor import PythonExecutor


class ScriptError(Exception):
    """Raised when script execution fails."""
    pass


class ScriptRunner:
    """
    Runs skill scripts in isolated IPython environments.

    Each script must define a main() function as its entrypoint.
    Scripts are executed in a separate IPython instance to avoid
    polluting the agent's runtime namespace.
    """

    async def run(
        self,
        script_path: Path,
        agent_runtime: PythonRuntime,
        **kwargs
    ) -> Any:
        """
        Execute script's main() in isolated runtime.

        Args:
            script_path: Path to the script file
            agent_runtime: PythonRuntime for scripts to access agent state/data
            **kwargs: Arguments passed to main()

        Returns:
            Return value from main()

        Raises:
            ScriptError: If script doesn't exist, lacks main(), or execution fails
        """
        if not script_path.exists():
            raise ScriptError(f"Script not found: '{script_path}'")

        # Create isolated executor
        executor = PythonExecutor()

        try:
            # Load script (defines main function)
            code = script_path.read_text(encoding="utf-8")
            result = await executor.execute(code)
            if not result.success:
                raise ScriptError(f"Failed to load script '{script_path.name}': {result.error}")

            # Get main function
            main_fn = executor.get_from_namespace("main")
            if main_fn is None:
                raise ScriptError(f"Script '{script_path.name}' must define a main() function")

            if not callable(main_fn):
                raise ScriptError(f"main in '{script_path.name}' is not callable")

            # Call main directly with runtime and kwargs
            try:
                if inspect.iscoroutinefunction(main_fn):
                    return await main_fn(runtime=agent_runtime, **kwargs)
                return main_fn(runtime=agent_runtime, **kwargs)
            except Exception as e:
                raise ScriptError(f"Script '{script_path.name}' execution failed: {e}")

        finally:
            executor.reset()
