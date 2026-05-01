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

    def cargar_muestras_temporales(self, archivo_muestras: Path) -> np.ndarray:
        """Carga una secuencia temporal binaria desde CSV (filas=tiempo, columnas=nodos)."""
        if not archivo_muestras.exists():
            raise FileNotFoundError(f"No se encontro el archivo de muestras: {archivo_muestras}")

        datos = np.genfromtxt(archivo_muestras, delimiter=CSV_DELIMITER)
        if datos.size == 0:
            raise ValueError(f"El archivo de muestras esta vacio: {archivo_muestras}")

        if datos.ndim == 1:
            if len(self.estado_inicial) == 1:
                datos = datos.reshape(-1, 1)
            else:
                datos = datos.reshape(1, -1)

        if datos.ndim != 2:
            raise ValueError(
                "Formato invalido de muestras: se esperaba una matriz 2D "
                "(filas=tiempo, columnas=nodos)."
            )

        if datos.shape[1] != len(self.estado_inicial):
            raise ValueError(
                "Columnas invalidas en muestras: "
                f"se esperaban {len(self.estado_inicial)} y llegaron {datos.shape[1]}."
            )

        if datos.shape[0] < 2:
            raise ValueError("Se requieren al menos 2 filas para estimar transiciones t->t+1.")

        if not np.isin(datos, [0, 1]).all():
            raise ValueError("Las muestras deben ser binarias (solo 0 y 1).")

        return datos.astype(np.int8, copy=False)

    def construir_tpm_desde_muestras(
        self,
        muestras: np.ndarray,
        valor_no_observado: float = 0.5,
    ) -> np.ndarray:
        """Construye TPM (2^n x n) estimando P(X_{t+1}=1 | estado_t) desde muestras."""
        if muestras.ndim != 2:
            raise ValueError("Las muestras deben ser una matriz 2D.")
        if muestras.shape[0] < 2:
            raise ValueError("Se requieren al menos 2 filas para construir la TPM.")
        if muestras.shape[1] != len(self.estado_inicial):
            raise ValueError(
                "Columnas invalidas en muestras: "
                f"se esperaban {len(self.estado_inicial)} y llegaron {muestras.shape[1]}."
            )
        if not np.isin(muestras, [0, 1]).all():
            raise ValueError("Las muestras deben ser binarias (solo 0 y 1).")

        num_nodos = muestras.shape[1]
        num_estados = 1 << num_nodos
        pesos = (1 << np.arange(num_nodos - 1, -1, -1)).astype(np.int64)

        estados_t = muestras[:-1].astype(np.int64, copy=False)
        estados_t1 = muestras[1:].astype(np.float32, copy=False)
        indices_t = (estados_t * pesos).sum(axis=1)

        conteos = np.bincount(indices_t, minlength=num_estados).astype(np.float32)
        acumulado_t1 = np.zeros((num_estados, num_nodos), dtype=np.float32)
        np.add.at(acumulado_t1, indices_t, estados_t1)

        tpm = np.full((num_estados, num_nodos), valor_no_observado, dtype=np.float32)
        observados = conteos > 0
        tpm[observados] = acumulado_t1[observados] / conteos[observados, None]
        return tpm

    def construir_tpm_desde_csv_muestras(
        self,
        archivo_muestras: Path,
        valor_no_observado: float = 0.5,
    ) -> np.ndarray:
        """Carga muestras temporales y devuelve la TPM estimada en formato 2^n x n."""
        muestras = self.cargar_muestras_temporales(archivo_muestras)
        return self.construir_tpm_desde_muestras(
            muestras,
            valor_no_observado=valor_no_observado,
        )

    def construir_tpm_bayesiana(
        self,
        muestras: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Estima la TPM con suavizado bayesiano usando un prior de Dirichlet.

        En lugar de usar frecuencias relativas puras (estimador de maxima
        verosimilitud), suma un pseudoconteo alpha a cada par estado->nodo.
        Esto evita probabilidades exactamente 0 o 1 cuando los datos son
        escasos, lo que hace la TPM mas robusta para calcular divergencias
        como Jensen-Shannon o KL que son indefinidas en esos extremos.

        alpha = 1.0  -> prior uniforme de Laplace (el mas neutral)
        alpha < 1.0  -> prior que concentra la masa en los extremos (mas informativo)
        alpha > 1.0  -> prior que suaviza mas agresivamente hacia 0.5
        """
        if muestras.ndim != 2:
            raise ValueError("Las muestras deben ser una matriz 2D.")
        if muestras.shape[0] < 2:
            raise ValueError("Se requieren al menos 2 filas.")
        if muestras.shape[1] != len(self.estado_inicial):
            raise ValueError(
                f"Columnas invalidas: se esperaban {len(self.estado_inicial)} "
                f"y llegaron {muestras.shape[1]}."
            )
        if not np.isin(muestras, [0, 1]).all():
            raise ValueError("Las muestras deben ser binarias (solo 0 y 1).")
        if alpha <= 0:
            raise ValueError(f"alpha debe ser positivo, se recibio {alpha}.")

        num_nodos = muestras.shape[1]
        num_estados = 1 << num_nodos
        pesos = (1 << np.arange(num_nodos - 1, -1, -1)).astype(np.int64)

        estados_t = muestras[:-1].astype(np.int64, copy=False)
        estados_t1 = muestras[1:].astype(np.float64, copy=False)
        indices_t = (estados_t * pesos).sum(axis=1)

        # Conteo de observaciones por estado + pseudoconteos de Dirichlet.
        conteos = np.bincount(indices_t, minlength=num_estados).astype(np.float64)
        acumulado_t1 = np.zeros((num_estados, num_nodos), dtype=np.float64)
        np.add.at(acumulado_t1, indices_t, estados_t1)

        # P(X_{t+1}=1 | estado) = (observaciones_1 + alpha) / (total + 2*alpha)
        # El denominador 2*alpha corresponde a una Dirichlet(alpha, alpha) sobre {0,1}.
        tpm = (acumulado_t1 + alpha) / (conteos[:, None] + 2.0 * alpha)
        return tpm.astype(np.float32)


# Alias retrocompatible.
Manager = Gestor
