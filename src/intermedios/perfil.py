from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional


class GestorPerfilado:
    """Gestor simple de perfilado con salida a archivos en review/profiling."""

    def __init__(self, enabled: bool = True, output_dir: Path = Path("review/profiling")):
        self.enabled = enabled
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sesion_actual: Optional[Path] = None

    def iniciar_sesion(self, nombre_sesion: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%d_%m_%Y/%Hhrs")
        sesion = self.output_dir / nombre_sesion / timestamp
        sesion.mkdir(parents=True, exist_ok=True)
        self.sesion_actual = sesion

    def obtener_ruta_salida(self, nombre: str, extension: str) -> Path:
        base = self.sesion_actual or (self.output_dir / "default")
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{nombre}.{extension}"

    # Aliases retrocompatibles.
    start_session = iniciar_sesion
    get_output_path = obtener_ruta_salida


gestor_perfilado = GestorPerfilado()


def perfilar(name: Optional[str] = None) -> Callable:
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

            nombre_perfil = name or func.__name__

            if profiler_available:
                profiler = Profiler(interval=0.001, async_mode="disabled")
                profiler.start()
                result = func(*args, **kwargs)
                profiler.stop()

                html_path = gestor_perfilado.obtener_ruta_salida(nombre_perfil, "html")
                html_path.write_text(
                    profiler.output(renderer=HTMLRenderer()),
                    encoding="utf-8",
                )
                return result

            import time

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            txt_path = gestor_perfilado.obtener_ruta_salida(nombre_perfil, "txt")
            txt_path.write_text(
                f"{nombre_perfil} elapsed_seconds={elapsed:.6f}\n",
                encoding="utf-8",
            )
            return result

        return wrapper

    return decorator


# Alias retrocompatible.
ProfilingManager = GestorPerfilado
profile = perfilar
