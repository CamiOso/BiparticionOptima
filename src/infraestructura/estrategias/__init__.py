"""Adaptadores de estrategia — implementaciones concretas de IEstrategia.

Cada clase expuesta aquí satisface el puerto ``IEstrategia`` y puede ser
inyectada en el caso de uso ``BuscarParticionOptima``.

Uso::

    from src.infraestructura.estrategias import FuerzaBruta, Geometric, QNodos, Phi

Estrategias disponibles
-----------------------
- FuerzaBruta  : enumeración exacta O(2^n), referencia de exactitud
- Phi          : integración con PyPhi o heurística de respaldo
- QNodos       : greedy submodular O(n²), rápido y aproximado
- Geometric    : recorrido hipercúbico O(n·2^n), modos STRICT/REFINED
- Circuito     : descomposición espectral del Laplaciano O(n³)
"""

from src.estrategias.fuerza_bruta import FuerzaBruta
from src.estrategias.phi import Phi
from src.estrategias.q_nodos import QNodos
from src.estrategias.geometrica import Geometric
from src.estrategias.circuito import Circuito
from src.estrategias.informacion_bottleneck import InformacionBottleneck
from src.estrategias.louvain import Louvain
from src.estrategias.genetico import AlgoritmoGenetico
from src.estrategias.particion_ilp import ParticionILP
from src.estrategias.belief_propagation import BeliefPropagation
from src.estrategias.remcmc import REMCMC
from src.estrategias.variacional import ParticionVariacional
from src.estrategias.branch_bound import BranchBound
from src.estrategias.hiperbolica import ParticionHiperbolica
from src.estrategias.ib_qnodos import IBQNodos

__all__ = [
    "FuerzaBruta",
    "Phi",
    "QNodos",
    "Geometric",
    "Circuito",
    "InformacionBottleneck",
    "Louvain",
    "AlgoritmoGenetico",
    "ParticionILP",
    "BeliefPropagation",
    "REMCMC",
    "ParticionVariacional",
    "BranchBound",
    "ParticionHiperbolica",
    "IBQNodos",
]
