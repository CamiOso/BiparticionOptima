"""Adaptadores de observabilidad — logging y profiling.

Implementaciones concretas de ``IRegistro`` y del gestor de perfilado.
Pueden ser sustituidas por cualquier otra implementación sin modificar
los casos de uso.

Uso::

    from src.infraestructura.observabilidad import SafeLogger, gestor_perfilado, perfilar

Adaptadores disponibles
-----------------------
- SafeLogger        : implementa IRegistro con salida a consola y archivo
- gestor_perfilado  : gestiona sesiones de profiling con pyinstrument
- perfilar          : decorador para medir tiempo de ejecución de funciones
"""

from src.intermedios.registro import SafeLogger
from src.intermedios.perfil import gestor_perfilado, perfilar

__all__ = ["SafeLogger", "gestor_perfilado", "perfilar"]
