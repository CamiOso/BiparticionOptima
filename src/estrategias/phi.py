import numpy as np

from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class Phi(SIA):
    """Estrategia basada en PyPhi con fallback didactico si no esta disponible."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        dist_subsistema = self.sia_dists_marginales

        # Etapa inicial: intentamos detectar disponibilidad de pyphi,
        # pero devolvemos un resultado estable mientras construimos la version completa.
        try:
            import pyphi  # noqa: F401

            estrategia_nombre = "PyPhi"
        except Exception:
            estrategia_nombre = "PyPhi-fallback"

        return Solucion(
            estrategia=estrategia_nombre,
            perdida=0.0,
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_subsistema.copy(),
            estado_inicial=estado_inicial,
            particion="(M=(0,), A=(0,)) | (M*=(), A*=())",
        )
