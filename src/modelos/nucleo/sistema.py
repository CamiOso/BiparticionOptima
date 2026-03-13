import numpy as np
from numpy.typing import NDArray

from src.funciones.iit import seleccionar_estado
from src.modelos.nucleo.ncubo import NCube


class Sistema:
    """Sistema compuesto por n-cubos derivados de una TPM."""

    def __init__(self, tpm: np.ndarray, estado_inicio: NDArray[np.int8]) -> None:
        self.estado_inicial = estado_inicio
        self.ncubos = self._crear_ncubos(tpm)
        self.memo: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[NCube, ...]] = {}

    @classmethod
    def _from_cubes(
        cls,
        estado_inicio: NDArray[np.int8],
        cubos: tuple[NCube, ...],
    ) -> "Sistema":
        instancia = cls.__new__(cls)
        instancia.estado_inicial = estado_inicio
        instancia.ncubos = cubos
        instancia.memo = {}
        return instancia

    def _crear_ncubos(self, tpm: np.ndarray) -> tuple[NCube, ...]:
        num_nodos = tpm.shape[1]
        filas_esperadas = 1 << num_nodos
        if tpm.shape[0] != filas_esperadas:
            raise ValueError(
                "TPM invalida: se esperaban "
                f"{filas_esperadas} filas para {num_nodos} nodos y llegaron {tpm.shape[0]}."
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
        return np.array([cubo.indice for cubo in self.ncubos], dtype=np.int8)

    @property
    def dims_ncubos(self) -> NDArray[np.int8]:
        if not self.ncubos:
            return np.array([], dtype=np.int8)
        return self.ncubos[0].dims

    def condicionar(self, indices: NDArray[np.int8]) -> "Sistema":
        """Aplica condiciones de fondo y elimina n-cubos de esos indices."""
        indices_en_sistema = np.intersect1d(self.indices_ncubos, indices)
        if not indices_en_sistema.size:
            return self

        nuevos_cubos = tuple(
            cubo.condicionar(indices_en_sistema, self.estado_inicial)
            for cubo in self.ncubos
            if cubo.indice not in indices_en_sistema
        )
        return Sistema._from_cubes(self.estado_inicial, nuevos_cubos)

    def substraer(
        self,
        alcance_idx: NDArray[np.int8],
        mecanismo_dims: NDArray[np.int8],
    ) -> "Sistema":
        """Remueve futuros (indices de n-cubo) y marginaliza dimensiones de mecanismo."""
        alcance_set = {int(v) for v in alcance_idx.tolist()}
        nuevos_cubos = tuple(
            cubo.marginalizar(mecanismo_dims)
            for cubo in self.ncubos
            if cubo.indice not in alcance_set
        )
        return Sistema._from_cubes(self.estado_inicial, nuevos_cubos)

    def bipartir(
        self,
        alcance_preservado: NDArray[np.int8],
        mecanismo_preservado: NDArray[np.int8],
    ) -> "Sistema":
        """Genera una biparticion replicando la semantica del sistema de referencia."""
        clave = (tuple(int(v) for v in alcance_preservado), tuple(int(v) for v in mecanismo_preservado))
        memo = getattr(self, "memo", {})
        if clave not in memo:
            memo[clave] = tuple(
                cubo.marginalizar(np.setdiff1d(cubo.dims, mecanismo_preservado))
                if cubo.indice in alcance_preservado
                else cubo.marginalizar(mecanismo_preservado)
                for cubo in self.ncubos
            )
        self.memo = memo
        return Sistema._from_cubes(self.estado_inicial, memo[clave])

    def distribucion_marginal(self) -> NDArray[np.float32]:
        """Calcula P(nodo_i = ON) en el estado inicial para cada n-cubo."""
        if not self.ncubos:
            return np.array([], dtype=np.float32)

        probabilidades = []
        for cubo in self.ncubos:
            probabilidad = cubo.data
            if cubo.dims.size:
                inicial = tuple(int(self.estado_inicial[int(dim)]) for dim in cubo.dims)
                probabilidad = cubo.data[tuple(seleccionar_estado(np.array(inicial, dtype=np.int8)).tolist())]
            probabilidades.append(float(probabilidad))
        return np.array(probabilidades, dtype=np.float32)


# Alias retrocompatible.
System = Sistema
