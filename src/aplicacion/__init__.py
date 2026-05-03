"""Capa de Aplicación — orquestación y contratos.

Contiene:
- configuracion : AppConfig (reemplaza al singleton mutable)
- puertos/      : interfaces (Protocols) que el dominio expone hacia la infraestructura
- casos_de_uso/ : orquestadores que coordinan entidades y puertos sin depender de adaptadores

Regla de dependencia: esta capa sólo puede importar desde src.dominio.
"""
