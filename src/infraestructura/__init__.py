"""Capa de Infraestructura — adaptadores que implementan los puertos.

Contiene todas las dependencias externas: algoritmos concretos, I/O de archivos,
logging, profiling y visualización. Cada subdirectorio agrupa adaptadores por tipo.

Subdirectorios
--------------
estrategias/    — implementaciones concretas de IEstrategia
repositorios/   — implementaciones concretas de IRepositorioTPM
observabilidad/ — implementaciones de IRegistro y perfilado
herramientas/   — benchmark y análisis espectral
visualizacion/  — rendering de particiones

Regla de dependencia: esta capa puede importar desde dominio y aplicacion,
pero nunca al revés.
"""
