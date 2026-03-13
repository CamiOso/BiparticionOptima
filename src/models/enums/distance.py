from enum import Enum


class MetricDistance(Enum):
    """Distancias disponibles para comparar distribuciones."""

    HAMMING = "distancia-hamming"
    MANHATTAN = "distancia-manhattan"
    EUCLIDIANA = "distancia-euclidiana"
