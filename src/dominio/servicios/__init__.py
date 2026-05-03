"""Servicios de dominio — algoritmos puros sin dependencias externas.

Re-exporta las funciones de cómputo central para que la capa de aplicación
las consuma a través de esta frontera.

Módulos disponibles
-------------------
- particiones   : generadores de biparticiones y k-particiones
- metricas_iit  : distancias EMD, Jensen-Shannon, Wasserstein, Fisher-Rao, KL
- entropia      : entropías de Shannon, Rényi y Tsallis
"""

from src.funciones.particiones import biparticiones, generar_candidatos, generar_subsistemas
from src.funciones.iit import (
    emd_efecto,
    emd_causal,
    jensen_shannon,
    kl_divergencia,
    wasserstein_sinkhorn,
    fisher_rao,
    seleccionar_emd,
)
from src.funciones.entropia import shannon, renyi, tsallis, perfil_entropia

__all__ = [
    "biparticiones",
    "generar_candidatos",
    "generar_subsistemas",
    "emd_efecto",
    "emd_causal",
    "jensen_shannon",
    "kl_divergencia",
    "wasserstein_sinkhorn",
    "fisher_rao",
    "seleccionar_emd",
    "shannon",
    "renyi",
    "tsallis",
    "perfil_entropia",
]
