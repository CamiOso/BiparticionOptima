# ProyectoAnalisis2026

Proyecto para construir, paso a paso, la logica de busqueda de biparticion optima.
## 1. Requisitos

- Python 3.11 o superior
- `pip`
- Git (opcional, para flujo de commits)

## 2. Clonar y entrar al proyecto

```bash
git clone https://github.com/CamiOso/BiparticionOptima.git
cd BiparticionOptima
```

Si ya estas en el workspace local, entra a:

```bash
cd /home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026
```

## 3. Crear y activar entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Nota:
- `requirements.txt` contiene lo minimo para correr pruebas (`pytest`).
- Si quieres forzar modo `PyPhi` en la estrategia `Phi`, instala `pyphi` manualmente.

## 5. Ejecutar el proyecto

```bash
python exec.py
```

Para ejecutar una estrategia especifica:

```bash
python exec.py --estrategia geometric --modo-geometric refinado
python exec.py --estrategia geometric --modo-geometric estricto
python exec.py --estrategia fuerza_bruta
python exec.py --estrategia phi
python exec.py --estrategia qnodos
```

Esto ejecuta `src/main.py` y muestra una demo de:
- `FuerzaBruta`
- `Phi` (PyPhi real si esta disponible, o heuristica)
- `QNodos` (version submodular con memoizacion)
- `Geometric` (busqueda sobre hipercubo con tabla de costos recursiva)

`Geometric` soporta dos modos:
- `estricto`: usa solo la tabla recursiva y seleccion geometrica base. Este es el modo que conserva la lectura teorica de complejidad `O(n·2^n)`.
- `refinado`: agrega refinamiento local y restarts adaptativos para mejorar precision empirica frente a `FuerzaBruta`.

## 6. Correr pruebas

```bash
PYTHONPATH=. python -m pytest -q
```

## 7. Benchmark de rendimiento (Geometric estricto/refinado vs FuerzaBruta)

```bash
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py
```

Genera un CSV en:

`review/benchmarks/geometric_vs_fuerza_bruta.csv`

con tiempos, speedup y diferencia de phi por corrida (multi-semilla) para:

- `FuerzaBruta`
- `Geometric` en modo `estricto`
- `Geometric` en modo `refinado`

Tambien genera:

`review/benchmarks/geometric_vs_fuerza_bruta_resumen.csv`

con promedio y mediana de speedup y `|delta phi|` por tamano de red.

## 8. Nota tecnica de complejidad

La justificacion formal del modo `estricto` y la distincion frente al modo `refinado` estan en:

`review/notas/complejidad_geometric.md`

## 9. Informe final de resultados

Resumen listo para entrega (metodologia, tablas y conclusiones):

`review/notas/informe_final_geometric.md`

## 10. Estructura principal

```text
src/
	constantes/      # Mensajes, etiquetas y configuracion base
	controladores/   # Carga de TPMs (CSV de muestras)
	funciones/       # Utilidades IIT, particiones y formato
	intermedios/     # Logging y perfilado
	modelos/         # Aplicacion, sistema, ncubo, solucion
	estrategias/     # FuerzaBruta, Phi, QNodos
	strategies/      # Geometric
	main.py          # Orquestador de ejecucion
exec.py            # Entry point
tests/             # Suite de pruebas
review/benchmarks/ # Scripts y salidas de benchmark
review/notas/      # Notas tecnicas e informe final
```

## 11. Flujo recomendado de trabajo

1. Crear/activar entorno virtual.
2. Instalar dependencias.
3. Ejecutar `python exec.py` para validacion rapida.
4. Ejecutar `PYTHONPATH=. python -m pytest -q` antes de cada commit.
5. Hacer cambios pequenos, validar, y luego commit/push.

## 12. Estado actual

- Carpeta y modulos en espanol.
- Estrategias funcionales con pruebas automatizadas.
- Estrategia `Geometric` integrada y benchmark reproducible.
- `Geometric` separado en modo `estricto` y `refinado`.
- `SIA` ya aplica `condicion`, `alcance` y `mecanismo` al preparar subsistema.
- `QNodos` ya usa una logica submodular con memoizacion.
- `Phi` usa `PyPhi` cuando esta disponible; si no, usa ruta heuristica.

## 13. Siguiente objetivo
- analisis completo de red en `FuerzaBruta` (candidatos/subsistemas/reporte),
- paridad avanzada de `Phi` (causa/efecto y repertorios),
- utilidades IIT restantes para equivalencia total.
