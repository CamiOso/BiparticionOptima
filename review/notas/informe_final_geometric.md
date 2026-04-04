# Informe final: Geometric (estricto/refinado) vs FuerzaBruta

## 1. Objetivo

Implementar y evaluar un algoritmo geometrico sobre hipercubos para encontrar biparticiones de minima perdida de informacion (MIP), contrastando:

- `FuerzaBruta` (referencia exacta, costo alto),
- `Geometric` modo `estricto` (enfoque teorico),
- `Geometric` modo `refinado` (enfoque practico de alta precision).

## 2. Alcance del resultado

Se completo lo siguiente:

- Estrategia `Geometric` implementada en `src/strategies/geometric.py`.
- Separacion formal de modos:
  - `estricto`
  - `refinado`
- Integracion en flujo principal (`src/main.py`).
- Tests automatizados para ambos modos.
- Benchmark comparativo multi-semilla contra `FuerzaBruta`.

## 3. Metodologia de evaluacion

### Configuracion de benchmark

Script:

`review/benchmarks/benchmark_geometric.py`

Entrada:

- nodos: 5, 6, 7, 8
- 3 semillas por cada tamano

Metricas:

- tiempo de ejecucion
- speedup frente a `FuerzaBruta`
- `|delta phi| = |phi_fuerza_bruta - phi_metodo|`

Archivos de salida:

- detalle por corrida: `review/benchmarks/geometric_vs_fuerza_bruta.csv`
- resumen agregado: `review/benchmarks/geometric_vs_fuerza_bruta_resumen.csv`

## 4. Resultados agregados

Fuente: `review/benchmarks/geometric_vs_fuerza_bruta_resumen.csv`

| nodos | speedup estricto (prom) | speedup refinado (prom) | delta phi estricto (prom) | delta phi refinado (prom) |
|---|---:|---:|---:|---:|
| 5 | 7.92x | 1.27x | 0.4492 | 0.0000 |
| 6 | 16.96x | 10.83x | 0.3961 | 0.0041 |
| 7 | 41.93x | 29.67x | 0.5977 | 0.0000 |
| 8 | 107.08x | 39.65x | 0.6166 | 0.0000 |

## 5. Interpretacion

1. `Geometric` modo `estricto`:
- maximiza velocidad,
- pero pierde mucha precision en `phi` respecto a `FuerzaBruta`.

2. `Geometric` modo `refinado`:
- mantiene speedup importante,
- recupera precision casi exacta respecto a `FuerzaBruta`.

3. Compromiso velocidad-precision:
- `estricto` es mejor para narrativa teorica y costos bajos,
- `refinado` es mejor para resultados cercanos/exactos en practica.

## 6. Complejidad teorica

La justificacion del modo teorico esta en:

`review/notas/complejidad_geometric.md`

Resumen:

- El modo `estricto` se presenta con lectura `O(n · 2^n)` sobre la parte base de tabla recursiva del hipercubo.
- El modo `refinado` agrega refinamientos (hill-climbing/adaptativo/restarts), por lo que no se presenta como estrictamente `O(n · 2^n)` en todos los casos.

## 7. Conclusiones finales

- El proyecto cumple con la implementacion de una estrategia geometrica funcional para MIP.
- Se dispone de un modo teorico (`estricto`) y uno practico (`refinado`).
- Frente a `FuerzaBruta`, el modo refinado logra:
  - alta aceleracion,
  - y precision muy alta en `phi` en la evaluacion multi-semilla reportada.

## 8. Reproducibilidad

Comandos:

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py
```

## 9. Cierre de pendientes de la guia paso a paso

Se incorporaron piezas faltantes para cubrir mejor el flujo metodologico completo:

1. Construccion de TPM desde muestras temporales (Paso 1)
- Ya se puede estimar una TPM `2^n x n` desde CSV temporal binario usando CLI:

```bash
python exec.py --estrategia geometric --estado-inicial 1000 --csv-muestras review/salidas/muestras_1000.csv
```

2. Ejemplo practico explicito con 3 variables (Paso 9)
- Se agrego script reproducible:

```bash
PYTHONPATH=. python review/benchmarks/ejemplo_3_variables.py
```

- El script realiza:
  - construccion/uso de TPM de 3 nodos,
  - calculo de costo para transicion `000 -> 011` via `gamma = 2^(-d)`,
  - busqueda de biparticion con `Geometric` (modo `estricto`),
  - export de tabla completa de costos entre estados del cubo de 3 variables.

- Salida generada:
  - `review/salidas/tabla_costos_3_variables.csv`

3. Limitaciones que se mantienen (Paso 10)
- El proyecto incluye refinamiento adaptativo y restarts en `Geometric`, pero no implementa aun:
  - paralelizacion explicita del calculo de costos dentro del solver,
  - reduccion por simetrias formales del hipercubo.

Estas dos optimizaciones quedan marcadas como trabajo futuro sin bloquear el flujo base de analisis ni la reproducibilidad actual.
