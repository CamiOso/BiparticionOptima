# ProyectoAnalisis2026

Implementación paso a paso de algoritmos para encontrar la **Partición de Mínima
Pérdida de Información (MIP)** en sistemas de nodos, en el contexto de la
Teoría de Información Integrada (IIT).

## ¿Qué hace este proyecto?

Dado un sistema de n nodos con una Matriz de Probabilidades de Transición (TPM),
encuentra la partición del sistema en grupos de tal forma que se pierda la menor
cantidad de información posible entre ellos.

Informe completo con ejemplos paso a paso: `review/notas/informe_explicado.md`

---

## Requisitos

- Python 3.11 o superior
- `pip`

## Instalación

```bash
git clone https://github.com/CamiOso/BiparticionOptima.git
cd BiparticionOptima
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Estrategias implementadas

| Estrategia    | Enfoque                                      | Complejidad   | Exacta |
|---------------|----------------------------------------------|---------------|--------|
| FuerzaBruta   | Prueba todas las particiones posibles         | O(2^n)        | Sí     |
| Phi           | PyPhi si disponible, heurística si no         | —             | Sí/No  |
| Geometric     | Búsqueda geométrica sobre hipercubo           | O(n·2^n)      | No     |
| QNodos        | Búsqueda submodular greedy con memoización    | O(n²)         | No     |
| Circuito      | Eigendescomposición del Laplaciano del grafo  | O(n³)         | No     |

`Geometric` tiene dos modos:
- `estricto`: solo tabla recursiva, mantiene la cota teórica `O(n·2^n)`.
- `refinado`: agrega hill-climbing y restarts para máxima precisión.

Todas las estrategias soportan **k-particiones** (k ≥ 2 grupos).

---

## Uso rápido

```bash
# Ejecutar todas las estrategias con la red de muestra
python exec.py

# Estrategia específica
python exec.py --estrategia geometric --modo-geometric refinado
python exec.py --estrategia geometric --modo-geometric estricto
python exec.py --estrategia fuerza_bruta
python exec.py --estrategia phi
python exec.py --estrategia qnodos

# Con estado inicial personalizado
python exec.py --estrategia geometric --estado-inicial 1000

# K-particiones (k grupos en vez de 2)
python exec.py --estrategia geometric --k-particiones 3

# Exportar resultado a JSON
python exec.py --estrategia geometric --modo-geometric refinado --output-json review/salidas/resultado.json

# Estimar TPM desde muestras temporales (CSV binario)
python exec.py --estrategia geometric --estado-inicial 1000 --csv-muestras review/salidas/muestras_1000.csv
```

### Usar la estrategia Circuito directamente

```python
from src.estrategias.circuito import Circuito
import numpy as np

tpm = np.random.rand(8, 3).astype(np.float32)
c = Circuito(tpm)

# Bipartición (k=2)
sol = c.aplicar_estrategia('101', '111', '111', '111', k=2)
print(sol)

# K-partición (k=3)
sol_k3 = c.aplicar_estrategia('101', '111', '111', '111', k=3)
print(sol_k3)
```

---

## Correr pruebas

```bash
PYTHONPATH=. python -m pytest -q

# Con cobertura (mínimo 70%)
PYTHONPATH=. pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

Referencia local (2026-04-04): `39 passed, 1 skipped`, cobertura `76.86%`.

---

## Benchmarks y ejemplos

### Geometric vs Fuerza Bruta

```bash
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py
```

Resultados reales (promedio de 3 semillas por tamaño):

| Nodos | Speedup Estricto | Speedup Refinado | Error φ Refinado |
|-------|-----------------|------------------|-----------------|
| 5     | 7.9x            | 1.3x             | 0.000           |
| 6     | 17.0x           | 10.8x            | 0.004           |
| 7     | 41.9x           | 29.7x            | 0.000           |
| 8     | 107.1x          | 39.7x            | 0.000           |

### Ejemplo guiado de 3 variables

```bash
PYTHONPATH=. python review/benchmarks/ejemplo_3_variables.py
```

Genera `review/salidas/tabla_costos_3_variables.csv` con costos `γ = 2^(-d)`
entre todos los pares de estados del cubo.

### Visualización del hipercubo (3 variables)

```bash
PYTHONPATH=. python review/benchmarks/visualizacion_3_variables.py
```

Genera:
- `review/salidas/hipercubo_3_variables.svg`
- `review/salidas/proyecciones_3_variables.csv`
- `review/salidas/adyacencia_hipercubo_3_variables.csv`

### Q-Nodos vs Geometric en k-particiones

```bash
PYTHONPATH=. python review/benchmarks/benchmark_k_particiones.py
```

Resultados (k=3, 5 semillas): Q-Nodos gana en precisión de φ en todos los casos;
Geometric gana en velocidad (hasta 48x más rápido para 4 nodos).

### Optimización para sistemas grandes (n ≥ 9)

```bash
PYTHONPATH=. python review/benchmarks/benchmark_geometric_optimizacion.py
```

| Nodos | Speedup con optimización | Error φ promedio |
|-------|--------------------------|-----------------|
| 9     | 1.05x                    | 0.006           |
| 10    | 1.22x                    | 0.000           |

---

## Estructura del proyecto

```
src/
  constantes/      # Etiquetas, mensajes y configuración base
  controladores/   # Carga de TPMs (desde CSV de muestras)
  funciones/       # Utilidades IIT, particiones y formato
  intermedios/     # Logging y perfilado
  modelos/         # Aplicacion, Sistema, NCube, Solucion
  estrategias/     # FuerzaBruta, Phi, QNodos, Circuito
  strategies/      # Geometric
  main.py          # Orquestador principal
exec.py            # Entry point CLI
tests/             # Suite de pruebas automatizadas
.github/workflows/ # CI (GitHub Actions)
review/
  benchmarks/      # Scripts de benchmark y CSVs de resultados
  salidas/         # Artefactos generados (CSVs, SVGs, JSONs)
  notas/           # Informes técnicos y análisis
```

---

## Documentación técnica

| Documento | Contenido |
|-----------|-----------|
| `review/notas/informe_explicado.md`    | Informe completo paso a paso, con ejemplos y tablas reales |
| `review/notas/informe_final_geometric.md` | Metodología y resultados de Geometric vs FuerzaBruta |
| `review/notas/complejidad_geometric.md`   | Justificación formal de la complejidad O(n·2^n)         |

---

## Muestras incluidas

`src/.samples/`: `N4A.csv`, `N5A.csv`, `N6A.csv`, `N7A.csv`, `N8A.csv`

---

## Flujo recomendado

```bash
source .venv/bin/activate
python exec.py                          # validación rápida
PYTHONPATH=. python -m pytest -q        # antes de cada commit
git add <archivos> && git commit -m "..."
git push origin main
```
