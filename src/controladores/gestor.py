from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.constantes.base import CSV_DELIMITER
from src.modelos.base.aplicacion import aplicacion


@dataclass
class Gestor:
    """Gestor de carga de TPMs desde muestras CSV."""

    estado_inicial: str
    ruta_base: Path = Path("src/.samples")

    @property
    def archivo_tpm(self) -> Path:
        nodos = len(self.estado_inicial)
        pagina = aplicacion.pagina_red_muestra
        return self.ruta_base / f"N{nodos}{pagina}.csv"

    @property
    def tpm_filename(self) -> Path:
        """Alias retrocompatible para codigo anterior."""
        return self.archivo_tpm

    def cargar_red(self) -> np.ndarray:
        if not self.archivo_tpm.exists():
            disponibles = sorted(p.name for p in self.ruta_base.glob("N*A.csv"))
            listado = ", ".join(disponibles) if disponibles else "(sin muestras disponibles)"
            raise FileNotFoundError(
                "No se encontro la muestra TPM esperada: "
                f"{self.archivo_tpm}. "
                f"Muestras disponibles en {self.ruta_base}: {listado}"
            )
        return np.genfromtxt(self.archivo_tpm, delimiter=CSV_DELIMITER)


# Alias retrocompatible.
Manager = Gestor
