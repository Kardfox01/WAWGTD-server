import time
from logging import getLogger
from functools import wraps
from rich.console import Console


console = Console()
getLogger("uvicorn.error").disabled = True

def LOG(message, condition=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not condition:
                return func(*args, **kwargs)

            with console.status(f"[bold cyan]{message}..."):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time
                    console.log(f"[bold cyan]{message} [green]OK[/] "
                                f"(время: {execution_time:.3f}с)")
                    return result
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    console.log(f"[bold cyan]{message} [red]СБОЙ[/] "
                                f"(время: {execution_time:.3f}с): {str(e)}")
                    raise
        return wrapper
    return decorator
