from pathlib import Path

import numpy as np
import pandas as pd

from src.funciones.particiones import (
    biparticiones,
    etiqueta_subconjunto,
    generar_candidatos,
    generar_subsistemas,
)
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import literales, seleccionar_emd
from src.intermedios.perfil import gestor_perfilado, perfilar
from src.intermedios.registro import SafeLogger
from src.modelos.base.sia import SIA
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion


class FuerzaBruta(SIA):
    """Implementacion inicial de fuerza bruta (version didactica minima)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self.logger = SafeLogger("bruteforce_strategy")
        gestor_perfilado.iniciar_sesion("FuerzaBruta")

    @perfilar(name="FuerzaBruta_aplicar_estrategia")
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
        self.logger.info("Iniciando estrategia FuerzaBruta.")
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        dist_subsistema = self.sia_dists_marginales

        assert self.sia_subsistema is not None
        alcance_indices = self.sia_subsistema.indices_ncubos
        mecanismo_indices = self.sia_subsistema.dims_ncubos

        mejor_perdida = np.inf
        mejor_dist_particion = dist_subsistema.copy()
        mejor_subalcance: tuple[int, ...] = ()
        mejor_submecanismo: tuple[int, ...] = ()

        for subalcance, submecanismo in biparticiones(alcance_indices, mecanismo_indices):
            sistema_partido = self.sia_subsistema.bipartir(
                np.array(subalcance, dtype=np.int8),
                np.array(submecanismo, dtype=np.int8),
            )
            dist_particion = sistema_partido.distribucion_marginal()

            if dist_particion.size != dist_subsistema.size:
                # Alineacion didactica: completamos con ceros para comparar en igual espacio.
                distribucion_alineada = np.zeros_like(dist_subsistema)
                distribucion_alineada[: dist_particion.size] = dist_particion
                dist_particion = distribucion_alineada

            perdida = self.distancia_metrica(dist_subsistema, dist_particion)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist_particion = dist_particion
                mejor_subalcance = subalcance
                mejor_submecanismo = submecanismo

        perdida = mejor_perdida
        dist_particion = mejor_dist_particion
        biparticion_fmt = fmt_biparticion(
            mejor_subalcance,
            mejor_submecanismo,
            tuple(int(v) for v in alcance_indices.tolist()),
            tuple(int(v) for v in mecanismo_indices.tolist()),
        )
        self.logger.debug(f"Perdida calculada: {perdida:.4f}")

        resultado = Solucion(
            estrategia="FuerzaBruta",
            perdida=float(perdida),
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_particion,
            estado_inicial=estado_inicial,
            particion=biparticion_fmt,
        )
        self.logger.info("Estrategia FuerzaBruta finalizada.")
        return resultado

    def analizar_red_completa(
        self,
        estado_inicial: str,
        directorio_salida: Path | None = None,
    ) -> Path:
        """Genera archivos Excel con el analisis por candidato y subsistema."""
        estado_vector = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
        sistema_completo = Sistema(self.tpm, estado_vector)
        total_nodos = estado_vector.size

        if directorio_salida is None:
            directorio_salida = (
                Path("review")
                / "resolver"
                / f"red_{total_nodos}{aplicacion.pagina_red_muestra}"
                / estado_inicial
            )
        directorio_salida.mkdir(parents=True, exist_ok=True)

        for indices_condicionados in generar_candidatos(total_nodos):
            candidato = sistema_completo.condicionar(
                np.array(indices_condicionados, dtype=np.int8)
            )
            nombre_candidato = literales(candidato.dims_ncubos)
            archivo_salida = directorio_salida / f"{nombre_candidato}.xlsx"

            with pd.ExcelWriter(archivo_salida, engine="openpyxl") as escritor:
                for alcance_removido, mecanismo_removido in generar_subsistemas(
                    candidato.dims_ncubos
                ):
                    if len(alcance_removido) == candidato.indices_ncubos.size:
                        continue

                    subsistema = candidato.substraer(
                        np.array(alcance_removido, dtype=np.int8),
                        np.array(mecanismo_removido, dtype=np.int8),
                    )
                    distribucion = subsistema.distribucion_marginal()
                    tabla = self._analizar_particiones_completas(
                        distribucion,
                        subsistema,
                    )
                    nombre_hoja = self._nombre_subsistema(
                        candidato,
                        alcance_removido,
                        mecanismo_removido,
                    )[:31]
                    tabla.to_excel(escritor, sheet_name=nombre_hoja)

        return directorio_salida

    def _analizar_particiones_completas(
        self,
        distribucion_subsistema: np.ndarray,
        subsistema: Sistema,
    ) -> pd.DataFrame:
        alcance_total = tuple(int(v) for v in subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in subsistema.dims_ncubos.tolist())

        columnas = [
            etiqueta_subconjunto(subalcance, alcance_total)
            for subalcance, _ in biparticiones(
                subsistema.indices_ncubos,
                np.array([], dtype=np.int8),
            )
        ]
        columnas = list(dict.fromkeys(columnas))

        filas = [
            etiqueta_subconjunto(submecanismo, mecanismo_total)
            for _, submecanismo in biparticiones(
                np.array([], dtype=np.int8),
                subsistema.dims_ncubos,
            )
        ]
        filas = list(dict.fromkeys(filas))

        resultados = pd.DataFrame(index=filas, columns=columnas, dtype=np.float32)

        for subalcance, submecanismo in biparticiones(
            subsistema.indices_ncubos,
            subsistema.dims_ncubos,
        ):
            sistema_partido = subsistema.bipartir(
                np.array(subalcance, dtype=np.int8),
                np.array(submecanismo, dtype=np.int8),
            )
            distribucion_particion = sistema_partido.distribucion_marginal()
            distribucion_particion = self._alinear_distribucion(
                distribucion_particion,
                distribucion_subsistema,
            )
            perdida = self.distancia_metrica(
                distribucion_subsistema,
                distribucion_particion,
            )

            fila = etiqueta_subconjunto(submecanismo, mecanismo_total)
            columna = etiqueta_subconjunto(subalcance, alcance_total)
            resultados.loc[fila, columna] = float(perdida)

        return resultados.fillna(np.nan)

    def _nombre_subsistema(
        self,
        candidato: Sistema,
        alcance_removido: tuple[int, ...],
        mecanismo_removido: tuple[int, ...],
    ) -> str:
        alcance_restante = np.setdiff1d(candidato.indices_ncubos, np.array(alcance_removido))
        mecanismo_restante = np.setdiff1d(
            candidato.dims_ncubos,
            np.array(mecanismo_removido),
        )
        return f"{literales(alcance_restante)}|{literales(mecanismo_restante, minuscula=True)}"

    def _alinear_distribucion(
        self,
        distribucion: np.ndarray,
        referencia: np.ndarray,
    ) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        distribucion_alineada = np.zeros_like(referencia)
        distribucion_alineada[: distribucion.size] = distribucion
        return distribucion_alineada


# Alias retrocompatible.
BruteForce = FuerzaBruta
