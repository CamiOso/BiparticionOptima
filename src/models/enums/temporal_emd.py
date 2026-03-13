from enum import Enum


class TimeEMD(Enum):
    """Modo temporal para evaluacion EMD."""

    EMD_EFECTO = "emd-effect"
    EMD_CAUSA = "emd-cause"
    EMD_INTEGRADA = "emd-cause-effect"
