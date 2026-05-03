"""Casos de uso de la aplicación.

Cada caso de uso orquesta entidades del dominio y puertos sin acoplarse
a ningún adaptador concreto. Recibe sus dependencias por constructor
(inyección de dependencias).

Casos de uso disponibles
------------------------
- BuscarParticionOptima : encuentra la partición que minimiza φ
- EstimarTPM            : carga o estima la TPM del sistema
"""
