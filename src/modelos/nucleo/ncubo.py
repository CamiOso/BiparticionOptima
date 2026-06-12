from dataclasses import dataclass, field
import threading

import numpy as np
from numpy.typing import NDArray

_MAX_MEMO_NCUBE = 64


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
        object.__setattr__(self, "_lock", threading.Lock())
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma invalida {self.data.shape} para dimensiones {tuple(self.dims)}"
            )

    def __getstate__(self) -> dict:
        return {"indice": self.indice, "dims": self.dims, "data": self.data, "memo": {}}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "indice", state["indice"])
        object.__setattr__(self, "dims", state["dims"])
        object.__setattr__(self, "data", state["data"])
        object.__setattr__(self, "memo", state["memo"])
        object.__setattr__(self, "_lock", threading.Lock())

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
        cached = self.memo.get(key)
        if cached is not None:
            return NCube(indice=self.indice, dims=cached[1], data=cached[0])

        # Sets de Python son 11-23× más rápidos que np.intersect1d/setdiff1d
        # para arrays pequeños (tamaño mec, típicamente 10-25 elementos)
        ejes_set = set(key)
        dims_list = [int(d) for d in self.dims]
        marginable_set = ejes_set.intersection(dims_list)
        if not marginable_set:
            return self

        numero_dims = len(dims_list) - 1
        ejes_locales = tuple(
            numero_dims - dim_idx
            for dim_idx, axis in enumerate(dims_list)
            if axis in marginable_set
        )
        new_dims = np.array(
            [d for d in dims_list if d not in marginable_set],
            dtype=np.int8,
        )
        data_marginal = np.mean(self.data, axis=ejes_locales, keepdims=False)

        with self._lock:
            if key not in self.memo:
                if len(self.memo) >= _MAX_MEMO_NCUBE:
                    for k in list(self.memo.keys())[: _MAX_MEMO_NCUBE // 2]:
                        del self.memo[k]
                self.memo[key] = (data_marginal, new_dims)

        return NCube(
            indice=self.indice,
            dims=self.memo[key][1],
            data=self.memo[key][0],
        )
