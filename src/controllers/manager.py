from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.constants.base import CSV_DELIMITER
from src.models.base.application import aplicacion


@dataclass
class Manager:
    """Gestor de carga de TPMs desde muestras CSV."""

    estado_inicial: str
    ruta_base: Path = Path("src/.samples")

    @property
    def tpm_filename(self) -> Path:
        nodos = len(self.estado_inicial)
        pagina = aplicacion.pagina_red_muestra
        return self.ruta_base / f"N{nodos}{pagina}.csv"

    def cargar_red(self) -> np.ndarray:
        return np.genfromtxt(self.tpm_filename, delimiter=CSV_DELIMITER)
