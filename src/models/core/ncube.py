from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NCube:
    """Representa un cubo n-dimensional con operaciones basicas."""

    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray
    memo: dict[tuple[int, ...], tuple[np.ndarray, NDArray[np.int8]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma invalida {self.data.shape} para dimensiones {tuple(self.dims)}"
            )

    def condicionar(
        self,
        indices_condicionados: NDArray[np.int8],
        estado_inicial: NDArray[np.int8],
    ) -> "NCube":
        numero_dims = self.dims.size
        seleccion = [slice(None)] * numero_dims

        for condicion in indices_condicionados:
            level_arr = numero_dims - (condicion + 1)
            seleccion[level_arr] = estado_inicial[condicion]

        nuevas_dims = np.array(
            [dim for dim in self.dims if dim not in indices_condicionados],
            dtype=np.int8,
        )

        return NCube(
            indice=self.indice,
            dims=nuevas_dims,
            data=self.data[tuple(seleccion)],
        )

    def marginalizar(self, ejes: NDArray[np.int8]) -> "NCube":
        key = tuple(int(v) for v in ejes)
        if key not in self.memo:
            marginable_axis = np.intersect1d(ejes, self.dims)
            if not marginable_axis.size:
                return self

            numero_dims = self.dims.size - 1
            ejes_locales = tuple(
                numero_dims - dim_idx
                for dim_idx, axis in enumerate(self.dims)
                if axis in marginable_axis
            )

            new_dims = np.array(
                [d for d in self.dims if d not in marginable_axis],
                dtype=np.int8,
            )

            data_marginal = np.mean(self.data, axis=ejes_locales, keepdims=False)
            self.memo[key] = (data_marginal, new_dims)

        return NCube(
            indice=self.indice,
            dims=self.memo[key][1],
            data=self.memo[key][0],
        )
