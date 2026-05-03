"""Puertos de la capa de Aplicación (Ports en arquitectura hexagonal).

Un puerto es un contrato (Protocol) que define lo que la aplicación
necesita del mundo exterior, sin saber cómo se implementa.

Puertos disponibles
-------------------
- IEstrategia     : cualquier algoritmo que resuelva la partición óptima
- IRepositorioTPM : cualquier fuente que entregue una TPM
- IRegistro       : cualquier mecanismo de logging
"""
