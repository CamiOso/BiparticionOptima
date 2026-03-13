from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional


class ProfilingManager:
    """Gestor simple de perfilado con salida a archivos en review/profiling."""

    def __init__(self, enabled: bool = True, output_dir: Path = Path("review/profiling")):
        self.enabled = enabled
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[Path] = None

    def start_session(self, session_name: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%d_%m_%Y/%Hhrs")
        session = self.output_dir / session_name / timestamp
        session.mkdir(parents=True, exist_ok=True)
        self.current_session = session

    def get_output_path(self, name: str, ext: str) -> Path:
        base = self.current_session or (self.output_dir / "default")
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{name}.{ext}"


gestor_perfilado = ProfilingManager()


def profile(name: Optional[str] = None) -> Callable:
    """Decorador de perfilado. Intenta usar pyinstrument; si no existe usa temporizador."""

    def decorator(func: Callable) -> Callable:
        profiler_available = True
        try:
            from pyinstrument import Profiler
            from pyinstrument.renderers import HTMLRenderer
        except Exception:
            profiler_available = False

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not gestor_perfilado.enabled:
                return func(*args, **kwargs)

            profile_name = name or func.__name__

            if profiler_available:
                profiler = Profiler(interval=0.001, async_mode="disabled")
                profiler.start()
                result = func(*args, **kwargs)
                profiler.stop()

                html_path = gestor_perfilado.get_output_path(profile_name, "html")
                html_path.write_text(
                    profiler.output(renderer=HTMLRenderer()),
                    encoding="utf-8",
                )
                return result

            import time

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            txt_path = gestor_perfilado.get_output_path(profile_name, "txt")
            txt_path.write_text(
                f"{profile_name} elapsed_seconds={elapsed:.6f}\n",
                encoding="utf-8",
            )
            return result

        return wrapper

    return decorator
