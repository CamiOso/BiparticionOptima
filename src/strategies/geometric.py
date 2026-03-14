from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constantes.models import GEOMETRIC_LABEL
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import biparticiones
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


@dataclass(frozen=True)
class _ResultadoParticion:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


class Geometric(SIA):
    """Estrategia geometrica sobre hipercubo para aproximar la MIP en O(n*2^n)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self._cache_particiones: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[float, np.ndarray],
        ] = {}

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self._cache_particiones.clear()
        _ = self._tpm_a_tensores_elementales()

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        if not alcance_total and not mecanismo_total:
            return Solucion(
                estrategia=GEOMETRIC_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        # En sistemas pequenos usamos solucion exacta para validar equivalencia con fuerza bruta.
        n_nodos = len(set(alcance_total) | set(mecanismo_total))
        if n_nodos <= 4:
            mejor = self._resolver_exacto(alcance_total, mecanismo_total)
        else:
            mejor = self._resolver_geometrico(alcance_total, mecanismo_total)

        return Solucion(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor.perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor.distribucion,
            estado_inicial=estado_inicial,
            particion=fmt_biparticion(
                mejor.subalcance,
                mejor.submecanismo,
                alcance_total,
                mecanismo_total,
            ),
        )

    def _tpm_a_tensores_elementales(self) -> tuple[np.ndarray, ...]:
        """Representa cada n-cubo del subsistema como tensor elemental."""
        assert self.sia_subsistema is not None
        return tuple(np.array(cubo.data, dtype=np.float32, copy=True) for cubo in self.sia_subsistema.ncubos)

    def _resolver_exacto(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        mejor = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )
        assert self.sia_subsistema is not None

        for subalcance, submecanismo in biparticiones(
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
        ):
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor.perdida:
                mejor = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )
        return mejor

    def _resolver_geometrico(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        nodos = sorted(set(alcance_total) | set(mecanismo_total))
        total_mascaras = 1 << len(nodos)

        costos = np.full(total_mascaras, np.inf, dtype=np.float64)
        costos_locales = np.full(total_mascaras, np.inf, dtype=np.float64)
        predecesor = np.full(total_mascaras, -1, dtype=np.int32)

        costos[0] = 0.0
        candidatos_costo_cero: set[int] = set()

        for mascara in range(1, total_mascaras):
            subalcance, submecanismo = self._particion_desde_mascara(
                mascara,
                nodos,
                alcance_total,
                mecanismo_total,
            )
            perdida_local, _ = self._evaluar_particion(subalcance, submecanismo)
            costos_locales[mascara] = perdida_local

            if perdida_local <= 1e-12 and mascara != (total_mascaras - 1):
                candidatos_costo_cero.add(mascara)

            bits = mascara
            while bits:
                lsb = bits & -bits
                anterior = mascara ^ lsb
                d = 1
                gamma = 2.0 ** (-d)
                costo = costos[anterior] + gamma * perdida_local
                if costo < costos[mascara]:
                    costos[mascara] = costo
                    predecesor[mascara] = anterior
                bits ^= lsb

        if not candidatos_costo_cero:
            # Si no hay costo cero, usamos las mascaras con mejor costo recursivo.
            mejor_costo = float(np.min(costos[1:]))
            candidatos_costo_cero = {
                mascara
                for mascara in range(1, total_mascaras - 1)
                if costos[mascara] <= mejor_costo + 1e-12
            }

        if not candidatos_costo_cero:
            candidatos_costo_cero = {int(np.argmin(costos_locales[1:])) + 1}

        mejor_resultado = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )

        for mascara in candidatos_costo_cero:
            subalcance, submecanismo = self._particion_desde_mascara(
                mascara,
                nodos,
                alcance_total,
                mecanismo_total,
            )
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor_resultado.perdida:
                mejor_resultado = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )

        return mejor_resultado

    def _particion_desde_mascara(
        self,
        mascara: int,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        seleccionados = {nodos[idx] for idx in range(len(nodos)) if mascara & (1 << idx)}
        subalcance = tuple(nodo for nodo in alcance_total if nodo in seleccionados)
        submecanismo = tuple(nodo for nodo in mecanismo_total if nodo in seleccionados)
        return subalcance, submecanismo

    def _evaluar_particion(
        self,
        subalcance: tuple[int, ...],
        submecanismo: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        clave = (subalcance, submecanismo)
        en_cache = self._cache_particiones.get(clave)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        sistema_partido = self.sia_subsistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8),
        )
        distribucion = sistema_partido.distribucion_marginal()
        distribucion = self._alinear_distribucion(distribucion, self.sia_dists_marginales)

        perdida = float(self.distancia_metrica(self.sia_dists_marginales, distribucion))
        resultado = (perdida, distribucion)
        self._cache_particiones[clave] = resultado
        return resultado

    def _alinear_distribucion(
        self,
        distribucion: np.ndarray,
        referencia: np.ndarray,
    ) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        salida = np.zeros_like(referencia)
        salida[: distribucion.size] = distribucion
        return salida


# Alias en espanol para conservar consistencia del proyecto.
Geometrica = Geometric
