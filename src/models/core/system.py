import numpy as np
from numpy.typing import NDArray

from src.models.core.ncube import NCube


class System:
    """Sistema compuesto por n-cubos derivados de una TPM."""

    def __init__(self, tpm: np.ndarray, estado_inicio: NDArray[np.int8]) -> None:
        self.estado_inicial = estado_inicio
        self.ncubos = self._crear_ncubos(tpm)

    def _crear_ncubos(self, tpm: np.ndarray) -> tuple[NCube, ...]:
        num_nodos = tpm.shape[1]
        expected_rows = 1 << num_nodos
        if tpm.shape[0] != expected_rows:
            raise ValueError(
                "TPM invalida: se esperaban "
                f"{expected_rows} filas para {num_nodos} nodos y llegaron {tpm.shape[0]}."
            )

        return tuple(
            NCube(
                indice=idx,
                dims=np.array(range(num_nodos), dtype=np.int8),
                data=tpm[:, idx].reshape((2,) * num_nodos),
            )
            for idx in range(num_nodos)
        )

    @property
    def indices_ncubos(self) -> NDArray[np.int8]:
        return np.array([cube.indice for cube in self.ncubos], dtype=np.int8)

    @property
    def dims_ncubos(self) -> NDArray[np.int8]:
        if not self.ncubos:
            return np.array([], dtype=np.int8)
        return self.ncubos[0].dims

    def distribucion_marginal(self) -> NDArray[np.float32]:
        """Calcula P(nodo_i = ON) en el estado inicial para cada n-cubo."""
        probs = []
        for cube in self.ncubos:
            seleccion = [slice(None)] * cube.dims.size
            for dim in cube.dims:
                posicion_local = cube.dims.size - (int(dim) + 1)
                seleccion[posicion_local] = int(self.estado_inicial[int(dim)])
            probs.append(float(cube.data[tuple(seleccion)]))
        return np.array(probs, dtype=np.float32)
