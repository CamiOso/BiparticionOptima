import numpy as np
from numpy.typing import NDArray

from src.modelos.nucleo.ncubo import NCube


class Sistema:
    """Sistema compuesto por n-cubos derivados de una TPM."""

    def __init__(self, tpm: np.ndarray, estado_inicio: NDArray[np.int8]) -> None:
        self.estado_inicial = estado_inicio
        self.ncubos = self._crear_ncubos(tpm)

    @classmethod
    def _from_cubes(
        cls,
        estado_inicio: NDArray[np.int8],
        cubes: tuple[NCube, ...],
    ) -> "Sistema":
        inst = cls.__new__(cls)
        inst.estado_inicial = estado_inicio
        inst.ncubos = cubes
        return inst

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

    def condicionar(self, indices: NDArray[np.int8]) -> "Sistema":
        """Aplica condiciones de fondo y elimina n-cubos de esos indices."""
        indices_validos = np.intersect1d(self.indices_ncubos, indices)
        if not indices_validos.size:
            return self

        nuevos = tuple(
            cube.condicionar(indices_validos, self.estado_inicial)
            for cube in self.ncubos
            if cube.indice not in indices_validos
        )
        return Sistema._from_cubes(self.estado_inicial, nuevos)

    def substraer(
        self,
        alcance_idx: NDArray[np.int8],
        mecanismo_dims: NDArray[np.int8],
    ) -> "Sistema":
        """Remueve futuros (indices de n-cubo) y marginaliza dimensiones de mecanismo."""
        alcance_set = {int(v) for v in alcance_idx.tolist()}
        nuevos = tuple(
            cube.marginalizar(mecanismo_dims)
            for cube in self.ncubos
            if cube.indice not in alcance_set
        )
        return Sistema._from_cubes(self.estado_inicial, nuevos)

    def bipartir(
        self,
        alcance_preservado: NDArray[np.int8],
        mecanismo_preservado: NDArray[np.int8],
    ) -> "Sistema":
        """Genera una particion preservando subset de alcance y mecanismo."""
        alcance_eliminar = np.setdiff1d(self.indices_ncubos, alcance_preservado)
        mecanismo_eliminar = np.setdiff1d(self.dims_ncubos, mecanismo_preservado)
        return self.substraer(alcance_eliminar, mecanismo_eliminar)

    def distribucion_marginal(self) -> NDArray[np.float32]:
        """Calcula P(nodo_i = ON) en el estado inicial para cada n-cubo."""
        if not self.ncubos:
            return np.array([], dtype=np.float32)

        probs = []
        for cube in self.ncubos:
            seleccion = [slice(None)] * cube.dims.size
            for local_idx, dim in enumerate(cube.dims):
                posicion_local = cube.dims.size - (local_idx + 1)
                seleccion[posicion_local] = int(self.estado_inicial[int(dim)])
            probs.append(float(cube.data[tuple(seleccion)]))
        return np.array(probs, dtype=np.float32)


# Alias retrocompatible.
System = Sistema
