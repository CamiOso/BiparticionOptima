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
- `requirements.txt` esta pinneado con versiones exactas para ejecucion reproducible.
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
python exec.py --estrategia geometric --estado-inicial 1000
python exec.py --estrategia geometric --modo-geometric refinado --output-json review/salidas/geometric_1000.json
python exec.py --estrategia geometric --estado-inicial 1000 --csv-muestras review/salidas/muestras_1000.csv
```

Notas de CLI:

- `--estado-inicial` define la cantidad de nodos (longitud del bitstring).
- La TPM esperada debe cumplir forma `2^n x n` para ese `n`.
- `--output-json` exporta resultados de la corrida en formato JSON.
- `--csv-muestras` permite estimar la TPM desde una secuencia temporal CSV binaria
	(filas=tiempo, columnas=nodos). Si se usa este flag, no se carga `src/.samples/N*A.csv`.
- El JSON incluye `elapsed_seconds` por estrategia ejecutada.

Si no existe el CSV esperado para el tamano solicitado, el CLI muestra un error claro con la ruta faltante y las muestras disponibles.

Muestras incluidas en `src/.samples/`:

- `N4A.csv`, `N5A.csv`, `N6A.csv`, `N7A.csv`, `N8A.csv`

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

Para medir cobertura localmente (mismo criterio que CI):

```bash
PYTHONPATH=. pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

Tambien se incluyen pruebas de CLI (`tests/test_cli.py`) para validar:

- seleccion de estrategia con `--estrategia`,
- seleccion de modo geometrico con `--modo-geometric`,
- manejo de argumentos invalidos.

CI valida automaticamente pruebas + cobertura minima del 70%.

Referencia local verificada (2026-04-04): `39 passed, 1 skipped` y cobertura total `76.86%` con el comando anterior.

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

## 8. Ejemplo guiado de 3 variables

Se incluye un ejemplo reproducible que cubre:

- TPM de 3 nodos (`2^3 x 3`)
- calculo de `gamma = 2^(-d)` para una transicion concreta (`000 -> 011`)
- ejecucion de `Geometric` para recuperar biparticion optima
- tabla de costos entre todos los pares de estados del cubo de 3 variables

Comando:

```bash
PYTHONPATH=. python review/benchmarks/ejemplo_3_variables.py
```

Salida principal generada:

`review/salidas/tabla_costos_3_variables.csv`

## 9. Nota tecnica de complejidad

La justificacion formal del modo `estricto` y la distincion frente al modo `refinado` estan en:

`review/notas/complejidad_geometric.md`

## 10. Informe final de resultados

Resumen listo para entrega (metodologia, tablas y conclusiones):

`review/notas/informe_final_geometric.md`

## 11. Estructura principal

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
.github/workflows/ # CI en GitHub Actions
review/benchmarks/ # Scripts y salidas de benchmark
review/notas/      # Notas tecnicas e informe final
```

## 12. Flujo recomendado de trabajo

1. Crear/activar entorno virtual.
2. Instalar dependencias.
3. Ejecutar `python exec.py` para validacion rapida.
4. Ejecutar `PYTHONPATH=. python -m pytest -q` antes de cada commit.
5. Hacer cambios pequenos, validar, y luego commit/push.

## 13. Estado actual

- Carpeta y modulos en espanol.
- Estrategias funcionales con pruebas automatizadas.
- Estrategia `Geometric` integrada y benchmark reproducible.
- `Geometric` separado en modo `estricto` y `refinado`.
- `Geometric` ya incorpora optimizacion para sistemas grandes (muestreo, simetrias y costos en paralelo).
- `SIA` ya aplica `condicion`, `alcance` y `mecanismo` al preparar subsistema.
- `Gestor` ya permite estimar TPM desde muestras temporales binarias (`--csv-muestras`).
- `QNodos` ya usa una logica submodular con memoizacion.
- `Phi` usa `PyPhi` cuando esta disponible; si no, usa ruta heuristica.

## 14. Comandos rapidos

```bash
python exec.py --estrategia geometric --modo-geometric refinado
python exec.py --estrategia geometric --estado-inicial 1000 --csv-muestras review/salidas/muestras_1000.csv
PYTHONPATH=. python review/benchmarks/ejemplo_3_variables.py
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py
```

## 15. Estado de cierre

Proyecto finalizado para el alcance definido:

- estrategias implementadas y funcionales,
- evaluacion experimental y reportes generados,
- documentacion tecnica y de resultados lista para entrega.
