"""Caso de uso: EstimarTPM.

Carga o estima la Transition Probability Matrix (TPM) del sistema.
No conoce ningún repositorio concreto; delega en el puerto
``IRepositorioTPM`` recibido por inyección de dependencias.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.aplicacion.puertos.registro import IRegistro
from src.aplicacion.puertos.repositorio_tpm import IRepositorioTPM


@dataclass
class EstimarTPM:
    """Orquesta la obtención de la TPM desde cualquier fuente.

    Atributos
    ---------
    repositorio : IRepositorioTPM
        Adaptador que implementa el acceso al almacenamiento.
    registro    : IRegistro
        Adaptador de logging para observabilidad.
    """

    repositorio: IRepositorioTPM
    registro: IRegistro

    def cargar_desde_muestra_predefinida(self) -> np.ndarray:
        """Devuelve la TPM para el tamaño y página configurados."""
        self.registro.info("Cargando TPM desde muestra predefinida")
        tpm = self.repositorio.cargar_red()
        self.registro.debug(f"TPM cargada con forma {tpm.shape}")
        return tpm

    def estimar_desde_csv(
        self,
        archivo: Path,
        valor_no_observado: float = 0.5,
    ) -> np.ndarray:
        """Estima la TPM a partir de una secuencia temporal en CSV."""
        self.registro.info(f"Estimando TPM desde muestras temporales: {archivo}")
        tpm = self.repositorio.construir_tpm_desde_csv_muestras(
            archivo,
            valor_no_observado=valor_no_observado,
        )
        self.registro.debug(f"TPM estimada con forma {tpm.shape}")
        return tpm
