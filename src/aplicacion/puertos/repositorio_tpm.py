"""Puerto de salida: contrato para cargar o estimar una Transition Probability Matrix.

Desacopla los casos de uso de cualquier detalle de almacenamiento:
CSV, base de datos, red, etc. La capa de aplicación sólo conoce este
contrato; la implementación concreta vive en infraestructura.
"""

from pathlib import Path
from typing import Protocol

import numpy as np


class IRepositorioTPM(Protocol):
    """Contrato para obtener una TPM [2^n × n] de cualquier fuente."""

    def cargar_red(self) -> np.ndarray:
        """Devuelve la TPM predefinida para el tamaño y página configurados."""
        ...

    def construir_tpm_desde_csv_muestras(
        self,
        archivo_muestras: Path,
        valor_no_observado: float = 0.5,
    ) -> np.ndarray:
        """Estima la TPM a partir de una secuencia temporal en CSV."""
        ...
