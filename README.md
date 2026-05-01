# ProyectoAnalisis2026

Implementación de algoritmos para encontrar la **Partición de Mínima Pérdida de
Información (MIP)** en sistemas de nodos, en el contexto de la Teoría de
Información Integrada (IIT).

## ¿Qué hace este proyecto?

Dado un sistema de n nodos con una Matriz de Probabilidades de Transición (TPM),
encuentra la partición del sistema en grupos de tal forma que se pierda la menor
cantidad de información posible entre ellos (φ mínimo).

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

| Estrategia    | Enfoque                                       | Complejidad   | Exacta |
|---------------|-----------------------------------------------|---------------|--------|
| FuerzaBruta   | Prueba todas las particiones posibles          | O(2^n)        | Sí     |
| Phi           | PyPhi si disponible, heurística si no          | —             | Sí/No  |
| Geometric     | Búsqueda geométrica sobre hipercubo            | O(n·2^n)      | No     |
| QNodos        | Búsqueda submodular greedy con memoización     | O(n²)         | No     |
| Circuito      | Eigendescomposición del Laplaciano del grafo   | O(n³)         | No     |

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

---

## Métricas de distancia disponibles

La distancia entre distribuciones se configura mediante `aplicacion.set_tiempo_emd()`:

| Métrica              | Enum                       | Descripción |
|----------------------|----------------------------|-------------|
| EMD efecto (default) | `TimeEMD.EMD_EFECTO`       | L1 simplificado: Σ\|u-v\| |
| Jensen-Shannon       | `TimeEMD.JENSEN_SHANNON`   | √(JS divergence), simétrica, acotada |
| KL divergencia       | `TimeEMD.KL_DIVERGENCIA`   | KL simétrica (u\|\|v + v\|\|u) / 2 |
| Wasserstein-Sinkhorn | `TimeEMD.WASSERSTEIN`      | Transporte óptimo regularizado |
| Fisher-Rao           | `TimeEMD.FISHER_RAO`       | Distancia geodésica en variedad estadística |

```python
from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.emd_temporal import TimeEMD

aplicacion.set_tiempo_emd(TimeEMD.JENSEN_SHANNON)
```

---

## Módulos de análisis matemático

### Entropías de orden superior

```python
from src.funciones.entropia import shannon, renyi, tsallis, perfil_entropia

p = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)

shannon(p)           # Entropía de Shannon H(X)
renyi(p, alpha=2.0)  # Entropía de Rényi H_α(X) — para α=2: información de colisión
tsallis(p, q=2.0)    # Entropía de Tsallis S_q(X) — no extensiva, útil en sistemas complejos
perfil_entropia(p)   # Tabla H_α para varios órdenes α (describe la forma de la distribución)
```

### Información mutua de orden superior

```python
from src.funciones.informacion_superior import o_information, matriz_dependencia

# O-information: mide si el sistema es redundante (Ω > 0) o sinérgico (Ω < 0)
resultado = o_information(tpm, estado_inicial)
print(resultado["o_information"])   # Ω
print(resultado["correlacion_total"])  # TC = Σ H(Xi) - H(X)
print(resultado["tipo"])            # "redundancia" | "sinergia" | "neutral"

# Matriz n×n de información mutua I(Xi; Xj) entre todos los pares
mat = matriz_dependencia(tpm, estado_inicial)
```

### Análisis espectral de la TPM

```python
from src.herramientas.espectral import analizar_tpm

resultado = analizar_tpm(tpm)
resultado.imprimir()

# Acceso directo a los valores
resultado.brecha_espectral        # gap = 1 - |λ₂| (velocidad de convergencia)
resultado.tiempo_mezcla_cota      # pasos para estar ε-cerca de la distribución límite
resultado.distribucion_estacionaria  # π tal que π P = π
resultado.es_ergodica             # True si todos los estados son accesibles
```

### Benchmark automatizado

```python
from src.herramientas.benchmark import Benchmark

bench = Benchmark(tpm, k=2)
resultado = bench.ejecutar(estados=["1000", "0100", "1100"])
resultado.imprimir()    # tabla con pérdida y tiempo por estrategia
resultado.resumen()     # estadísticas agregadas por estrategia
```

### Visualización de particiones

```python
from src.visualizacion.particion import (
    dibujar_biparticion,
    dibujar_k_particion,
    dibujar_comparacion_perdidas,
)

# Bipartición como grafo bipartito mecanismo → alcance
dibujar_biparticion(
    subalcance=(0, 1), submecanismo=(0,),
    alcance_total=(0, 1, 2), mecanismo_total=(0, 1, 2),
    perdida=0.25,
    guardar_en="review/salidas/biparticion.png",
)

# K-partición con nodos coloreados por grupo
dibujar_k_particion(
    nodos=[0, 1, 2, 3], asignacion=(0, 0, 1, 1),
    alcance_total=(0, 1, 2, 3), mecanismo_total=(0, 1, 2, 3),
    perdida=0.125,
    guardar_en="review/salidas/k_particion.png",
)

# Gráfica de barras comparando pérdidas entre estrategias
dibujar_comparacion_perdidas(
    {"FuerzaBruta": 0.25, "QNodos": 0.25, "Geometric": 0.25},
    guardar_en="review/salidas/comparacion.png",
)
```

### Estimación bayesiana de la TPM

```python
from src.controladores.gestor import Gestor

gestor = Gestor(estado_inicial="1000")
muestras = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.int8)

# Estimación frecuentista (máxima verosimilitud)
tpm_ml = gestor.construir_tpm_desde_muestras(muestras)

# Estimación bayesiana con prior de Dirichlet (alpha=1.0: prior de Laplace)
tpm_bayes = gestor.construir_tpm_bayesiana(muestras, alpha=1.0)
# alpha pequeño → confía más en los datos; alpha grande → suaviza hacia 0.5
```

---

## Arquitectura interna: patrón Template Method en k-particiones

La búsqueda de k-particiones está centralizada en `BuscadorKParticion` (Template Method).
La clase base define el algoritmo completo (búsqueda exacta para sistemas pequeños,
búsqueda local con restarts para sistemas grandes). Cada estrategia implementa
solo `evaluar_asignacion()`, que es la única parte que difiere entre ellas.

```
BuscadorKParticion  (abstracta)
├── buscar()             — decide exacto vs local según tamaño del sistema
├── _buscar_exacto()     — enumeración exhaustiva (umbral configurable)
├── _buscar_local()      — hill-climbing + restarts aleatorios
├── refinar_local()      — descenso por vecindad en el espacio de asignaciones
└── evaluar_asignacion() — abstracto: cada estrategia lo implementa distinto

_BuscadorKGeometric  → usa k_bipartir      (nodos espaciales: list[int])
_BuscadorKQNodos     → usa k_bipartir_temporal (vértices temporales: list[tuple])
BuscadorKRecocido    → recocido simulado (acepta peores soluciones con prob. e^{-Δ/T})
```

---

## Correr pruebas

```bash
PYTHONPATH=. python -m pytest -q

# Con cobertura (mínimo 70%)
PYTHONPATH=. pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

## Benchmarks y ejemplos

### Geometric vs Fuerza Bruta

```bash
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py
```

| Nodos | Speedup Estricto | Speedup Refinado | Error φ Refinado |
|-------|-----------------|------------------|-----------------|
| 5     | 7.9×            | 1.3×             | 0.000           |
| 6     | 17.0×           | 10.8×            | 0.004           |
| 7     | 41.9×           | 29.7×            | 0.000           |
| 8     | 107.1×          | 39.7×            | 0.000           |

### Q-Nodos vs Geometric en k-particiones

```bash
PYTHONPATH=. python review/benchmarks/benchmark_k_particiones.py
```

| Nodos | Speedup Geometric (k=3) | Quién gana en φ |
|-------|------------------------|-----------------|
| 4     | 48×                    | Q-Nodos         |
| 5     | 22×                    | Q-Nodos         |
| 6     | 10×                    | Q-Nodos         |

---

## Estructura del proyecto

```
src/
  constantes/         # Etiquetas, mensajes y configuración base
  controladores/      # Gestor: carga y estimación (ML y bayesiana) de TPMs
  funciones/
    iit.py            # EMD, Jensen-Shannon, KL, Wasserstein, Fisher-Rao
    entropia.py       # Shannon, Rényi, Tsallis, perfil de entropías
    informacion_superior.py  # O-information, correlación total, matriz de dependencia
    k_particion_buscador.py  # BuscadorKParticion, BuscadorKRecocido (SA)
    particiones.py    # Generadores de biparticiones y k-particiones
    formato.py        # Renderizado de soluciones y particiones
  intermedios/        # Logging y perfilado
  modelos/            # Aplicacion (singleton), Sistema, NCube, Solucion
  estrategias/        # FuerzaBruta, Phi, QNodos, Circuito
  strategies/         # Geometric
  visualizacion/
    particion.py      # Gráficas de bipartición, k-partición y comparación
  herramientas/
    benchmark.py      # Benchmark comparativo automático
    espectral.py      # Análisis espectral: eigenvalores, distribución estacionaria
  main.py             # Orquestador principal
exec.py               # Entry point CLI
tests/                # Suite de pruebas automatizadas
.github/workflows/    # CI (GitHub Actions)
review/
  benchmarks/         # Scripts de benchmark y CSVs de resultados
  salidas/            # Artefactos generados (CSVs, SVGs, JSONs, PNGs)
  notas/              # Informes técnicos y bitácoras
```

---

## Documentación técnica

| Documento | Contenido |
|-----------|-----------|
| `review/notas/informe_explicado.md`       | Informe completo paso a paso con tablas reales |
| `review/notas/informe_final_geometric.md` | Metodología y resultados Geometric vs FuerzaBruta |
| `review/notas/complejidad_geometric.md`   | Justificación formal de la complejidad O(n·2^n) |
| `review/notas/bitacora_k_particiones.md`  | Desarrollo de k-particiones, Circuito y nuevas herramientas |

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
