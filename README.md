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

### Estrategias clásicas

| Estrategia    | Enfoque                                       | Complejidad k=2 | Complejidad k>2            | Exacta |
|---------------|-----------------------------------------------|-----------------|----------------------------|--------|
| FuerzaBruta   | Prueba todas las particiones posibles          | O(2^n)          | —                          | Sí     |
| Phi           | PyPhi si disponible, heurística si no          | —               | —                          | Sí/No  |
| Geometric     | Búsqueda geométrica sobre hipercubo            | O(n·2^n)        | DP O(3^n·k) + SA           | No     |
| QNodos        | Búsqueda submodular greedy con memoización     | O(n²)           | Recursión Q + SA           | No     |
| Circuito      | Eigendescomposición del Laplaciano del grafo   | O(n³)           | O(n³) + k-means            | No     |

`Geometric` tiene dos modos para k = 2:
- `estricto`: solo tabla recursiva, mantiene la cota teórica `O(n·2^n)`.
- `refinado`: agrega hill-climbing y restarts para máxima precisión.

Para k > 2, **Geometric** usa DP de subconjuntos (`BuscadorKDP`) inicializado con los
costos del hipercubo ya calculados + recocido simulado. **Q-Nodos** usa partición
recursiva con memoización DP sobre `algoritmo_q` como warm-start + recocido simulado.

### Estrategias avanzadas de k-partición

| Estrategia             | Técnica central                                     | Complejidad           |
|------------------------|-----------------------------------------------------|-----------------------|
| InformacionBottleneck  | Minimización alternada IB (Tishby et al., 1999)     | O(n²·k·iter)          |
| Louvain                | Maximización de modularidad en grafo de acoplamientos | O(n²·iter)          |
| AlgoritmoGenetico      | Metaheurística evolutiva: torneo + cruce uniforme   | O(gen·pop·eval)       |
| ParticionILP           | Relajación LP del k-cut mínimo (solver HiGHS)       | O((n²k)³) → práctico |
| BeliefPropagation      | Loopy Belief Propagation con modelo de Potts        | O(iter·|E|·k²)        |
| REMCMC                 | Replica Exchange MCMC (Parallel Tempering)          | O(replicas·pasos)     |

Todas las estrategias soportan **k-particiones** (k ≥ 2 grupos).

### Estrategias de inspiración matemática avanzada

Gracias a la orientación de matemáticos de la Universidad Nacional, se plantearon tres estrategias adicionales que exploran conexiones con áreas de matemáticas que van más allá del alcance habitual del problema MIP. Se mencionan aquí porque son parte del repositorio y resultan conceptualmente interesantes, aunque por la profundidad de las teorías que las sustentan no se entra en detalles:

- **ParticionVariacional** — basada en el operador de Schrödinger con potencial tipo Airy y en el Laplaciano normalizado espectral. La idea parte de que la frontera de la partición puede verse como el "turning point" de una función de Airy, en analogía con mecánica cuántica.

- **BranchBound** — búsqueda exacta para sistemas pequeños y heurística multi-arranque con expansión de vecindad de Hamming para sistemas grandes. Garantiza el óptimo cuando el número total de bits del subsistema es manejable.

- **ParticionHiperbolica** — inspirada en la fórmula de Ryu-Takayanagi (AdS/CFT, 2006), que establece una equivalencia entre entropía de entrelazamiento cuántico y área de superficies geodésicas en espacios hiperbólicos. Los nodos se proyectan al disco de Poincaré y la partición se busca como la geodésica de menor "costo". Hay conexión formal con la Relatividad General a través de la dualidad espacio Anti-de Sitter / teoría de campos conforme.

### Infraestructura de búsqueda de k-particiones

```
BuscadorKParticion  (Template Method — greedy local)
├── BuscadorKRecocido   (Simulated Annealing)
│   └── BuscadorKDP     (DP de subconjuntos O(3^n·k) + SA)
```

`BuscadorKDP` acepta costos precalculados (`costos_subconjuntos`) para evitar
recomputo cuando la estrategia ya los tiene disponibles (caso Geometric).

---

## Uso rápido

```bash
# Ejecutar todas las estrategias clásicas con la red de muestra
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

### Usar las estrategias avanzadas directamente

```python
import numpy as np
from src.estrategias.informacion_bottleneck import InformacionBottleneck
from src.estrategias.louvain import Louvain
from src.estrategias.genetico import AlgoritmoGenetico
from src.estrategias.particion_ilp import ParticionILP
from src.estrategias.belief_propagation import BeliefPropagation

# Cualquier TPM válida (2^n filas, n columnas)
tpm = np.random.rand(16, 4).astype(np.float32)
estado, mascara = "1000", "1111"

for Cls in [InformacionBottleneck, Louvain, AlgoritmoGenetico, ParticionILP, BeliefPropagation]:
    sol = Cls(tpm).aplicar_estrategia(estado, mascara, mascara, mascara, k=2)
    print(f"{sol.estrategia:28s}  φ={sol.perdida:.4f}  {sol.particion}")
```

---

## Arquitectura hexagonal (clean architecture)

El proyecto está organizado en cuatro capas con dependencias unidireccionales:

```
Dominio ← Aplicacion ← Infraestructura ← Presentacion
```

| Capa | Directorio | Contenido |
|------|-----------|-----------|
| Dominio | `src/dominio/` | Entidades puras (NCube, Sistema, Solucion), enumeraciones, servicios (métricas, particiones) |
| Aplicación | `src/aplicacion/` | `AppConfig` (reemplaza singleton), puertos `IEstrategia`/`IRepositorioTPM`/`IRegistro`, casos de uso |
| Infraestructura | `src/infraestructura/` | Adaptadores concretos: estrategias, repositorios CSV, logging, visualización |
| Presentación | `src/presentacion/` | `orquestador.py` — punto de entrada con inyección de dependencias |

El **composition root** (`src/contenedor.py`) es el único lugar donde se ensamblan los adaptadores:

```python
from src.contenedor import Contenedor
from src.aplicacion.configuracion import AppConfig
from src.aplicacion.casos_de_uso.buscar_particion import EntradaBusqueda
from src.dominio.enumeraciones import TimeEMD

# Configuración inmutable inyectada (sin singleton global)
config = AppConfig(tiempo_emd=TimeEMD.JENSEN_SHANNON.value)
contenedor = Contenedor(config)

# Caso de uso con dependencias inyectadas
caso = contenedor.caso_uso_buscar_particion("louvain", tpm)
resultado = caso.ejecutar(EntradaBusqueda("1000", "1111", "1111", "1111", k=3))
print(resultado.perdida)
```

O usando el orquestador de presentación:

```python
from src.presentacion.orquestador import ejecutar

resultado = ejecutar(
    estrategia="ib",          # o "louvain", "genetico", "ilp", "bp"
    estado_inicial="1000",
    k_particiones=3,
    config=AppConfig(tiempo_emd=TimeEMD.WASSERSTEIN.value),
)
```

---

## Métricas de distancia disponibles

La distancia entre distribuciones se configura mediante `AppConfig` o el singleton `aplicacion`:

| Métrica              | Enum                       | Descripción |
|----------------------|----------------------------|-------------|
| EMD efecto (default) | `TimeEMD.EMD_EFECTO`       | L1 simplificado: Σ\|u-v\| |
| Jensen-Shannon       | `TimeEMD.JENSEN_SHANNON`   | √(JS divergence), simétrica, acotada |
| KL divergencia       | `TimeEMD.KL_DIVERGENCIA`   | KL simétrica (u\|\|v + v\|\|u) / 2 |
| Wasserstein-Sinkhorn | `TimeEMD.WASSERSTEIN`      | Transporte óptimo regularizado |
| Fisher-Rao           | `TimeEMD.FISHER_RAO`       | Distancia geodésica en variedad estadística |

```python
# Opción 1: inyección limpia (nueva arquitectura)
from src.aplicacion.configuracion import AppConfig
from src.dominio.enumeraciones import TimeEMD

config = AppConfig(tiempo_emd=TimeEMD.JENSEN_SHANNON.value)
solver = AlgoritmoGenetico(tpm, config=config)

# Opción 2: singleton global (compatible con código existente)
from src.modelos.base.aplicacion import aplicacion
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

### Grafo de acoplamientos (compartido por Louvain, ILP y BP)

```python
from src.funciones.grafo_info import construir_afinidad

nodos, W = construir_afinidad(subsistema)
# W[i][j] = sensibilidad promedio del nodo i a cambios en el nodo j
# Base matemática compartida entre Louvain, ParticionILP y BeliefPropagation
```

### Análisis espectral de la TPM

```python
from src.herramientas.espectral import analizar_tpm

resultado = analizar_tpm(tpm)
resultado.imprimir()

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

dibujar_biparticion(
    subalcance=(0, 1), submecanismo=(0,),
    alcance_total=(0, 1, 2), mecanismo_total=(0, 1, 2),
    perdida=0.25,
    guardar_en="review/salidas/biparticion.png",
)

dibujar_k_particion(
    nodos=[0, 1, 2, 3], asignacion=(0, 0, 1, 1),
    alcance_total=(0, 1, 2, 3), mecanismo_total=(0, 1, 2, 3),
    perdida=0.125,
    guardar_en="review/salidas/k_particion.png",
)

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

tpm_ml    = gestor.construir_tpm_desde_muestras(muestras)
tpm_bayes = gestor.construir_tpm_bayesiana(muestras, alpha=1.0)
```

---

## Correr pruebas

```bash
PYTHONPATH=. python -m pytest -q

# Con cobertura (mínimo 70%)
PYTHONPATH=. pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

## Benchmarks

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

| Nodos | Speedup Geometric (k=3) | Quién gana en φ |
|-------|------------------------|-----------------|
| 4     | 48×                    | Q-Nodos         |
| 5     | 22×                    | Q-Nodos         |
| 6     | 10×                    | Q-Nodos         |

### Estrategias ML vs QNodos (k=2, φ exacto)

Benchmark sobre sistemas reales N10A, N15B y N20A comparando QNodos contra las
estrategias ML con multi-start steepest descent paralelizado (`ThreadPoolExecutor`).
Todas las estrategias minimizan directamente el EMD (φ real), no un sustituto.
Resultados en `DatosML2026.xlsx`.

| Sistema | n | QNodos (s) | IB (s) | IBQNodos (s) | Speedup IBQNodos | φ idéntico |
|---------|---|-----------|--------|--------------|-----------------|------------|
| N10A    | 10 | 3.52     | 1.50   | **0.67**     | ×5              | ✓          |
| N15B    | 15 | 154 / 65 / 49 | 6.8 / 4.1 / 4.4 | **3.7 / 0.8 / 0.7** | ×41–82 | ✓ |
| N20A    | 20 | 26 899 / 10 712 / 11 263 | 803 / 199 / 227 | **93 / 28 / 31** | ×290–385 | ✓ |

**IBQNodos** (IB seed + QNodos SA refinement) es la estrategia más rápida en todos los
sistemas probados. En n=20 completa el caso más difícil en **92 segundos** frente a
las 7.5 horas de QNodos (×290 speedup), manteniendo φ exacto.

> A n≥25, QNodos es computacionalmente inviable (días/semanas). IBQNodos escala
> a n=22 en minutos con el hardware de escritorio descrito en la hoja `plataformas`.

---

## Estructura del proyecto

```
src/
  dominio/                  # Capa de Dominio (pura, sin deps externas)
    entidades/              → NCube, Sistema, Solucion
    enumeraciones/          → MetricDistance, TimeEMD, GeometricMode, Notation
    servicios/              → biparticiones, métricas EMD, entropías
  aplicacion/               # Capa de Aplicación
    configuracion.py        → AppConfig (reemplaza singleton mutable)
    puertos/                → IEstrategia, IRepositorioTPM, IRegistro (Protocols)
    casos_de_uso/           → BuscarParticionOptima, EstimarTPM
  infraestructura/          # Capa de Infraestructura (adaptadores)
    estrategias/            → todas las estrategias
    repositorios/           → Gestor (CSV)
    observabilidad/         → SafeLogger, perfilado
  presentacion/             # Capa de Presentación
    orquestador.py          → ejecutar() con inyección de dependencias
  contenedor.py             # Composition root: ensambla puertos con adaptadores
  constantes/               # Etiquetas, mensajes y configuración base
  controladores/            # Gestor: carga y estimación de TPMs
  estrategias/
    fuerza_bruta.py         → FuerzaBruta — enumeración exacta
    phi.py                  → Phi — PyPhi o heurística
    q_nodos.py              → QNodos — greedy submodular
    circuito.py             → Circuito — eigendescomposición Laplaciana
    informacion_bottleneck.py → InformacionBottleneck — minimización alternada IB
    louvain.py              → Louvain — modularidad en grafo de acoplamientos
    genetico.py             → AlgoritmoGenetico — metaheurística evolutiva
    particion_ilp.py        → ParticionILP — relajación LP del k-cut
    belief_propagation.py   → BeliefPropagation — LBP con modelo de Potts
  strategies/
    geometric.py            → Geometric — búsqueda sobre hipercubo
  funciones/
    iit.py                  → EMD, Jensen-Shannon, KL, Wasserstein, Fisher-Rao
    entropia.py             → Shannon, Rényi, Tsallis, perfil de entropías
    informacion_superior.py → O-information, correlación total, matriz de dependencia
    k_particion_buscador.py → BuscadorKParticion, BuscadorKRecocido (SA)
    particiones.py          → Generadores de biparticiones y k-particiones
    grafo_info.py           → construir_afinidad() — matriz W compartida
    formato.py              → Renderizado de soluciones y particiones
  intermedios/              # Logging y perfilado
  modelos/                  # Application (singleton), Sistema, NCube, Solucion
  visualizacion/
    particion.py            → Gráficas de bipartición, k-partición y comparación
  herramientas/
    benchmark.py            → Benchmark comparativo automático
    espectral.py            → Análisis espectral: eigenvalores, distribución estacionaria
  main.py                   # Orquestador original (retrocompatible)
exec.py                     # Entry point CLI
tests/                      # Suite de pruebas automatizadas
.github/workflows/          # CI (GitHub Actions)
review/
  benchmarks/               # Scripts de benchmark y CSVs de resultados
  salidas/                  # Artefactos generados (CSVs, SVGs, JSONs, PNGs)
  notas/                    # Informes técnicos y bitácoras
```

---

## Documentación técnica

| Documento | Contenido |
|-----------|-----------|
| `review/notas/informe_explicado.md`       | Informe completo paso a paso con tablas reales |
| `review/notas/informe_final_geometric.md` | Metodología y resultados Geometric vs FuerzaBruta |
| `review/notas/complejidad_geometric.md`   | Justificación formal de la complejidad O(n·2^n) |
| `review/notas/bitacora_k_particiones.md`  | Desarrollo completo: k-particiones, Circuito, arquitectura hexagonal y nuevas estrategias |

---

## Pruebas experimentales a gran escala

Experimentos sobre sistemas de **20, 22 y 25 nodos** usando las estrategias **QNodos** y **Geometric** para k = 2, 3, 4, 5. Los resultados se guardan en `DatosPruebas2026_1.xlsx` (hojas `20A-Elementos`, `22A-Elementos`, `25A-Elementos`).

### Estado actual (mayo 2026)

| Sistema | Estrategia | Filas completadas | Observaciones |
|---------|------------|:-----------------:|---------------|
| 20 nodos | QNodos     | 49 / 50           | Completado (fila 6 fue la última) |
| 20 nodos | Geometric  | ~10 / 50          | Filas n_max≤17 completas; n_max=19–20 excluidas por restricción de tiempo |
| 22 nodos | QNodos     | ~40 / 50          | En curso; filas n_max=21 toman 7–10h/k |
| 22 nodos | Geometric  | 10 / 50           | Bloqueada en fila 55 (n_max=19, k=3 ~55h) |
| 25 nodos | QNodos     |  0 / 7 (selección) | Selección representativa encadenada; arranca al terminar 22A |
| 25 nodos | Geometric  |  0 / 3 (selección) | Selección representativa encadenada; arranca al terminar 22A Geo |

> Las pruebas de 25A usan una **selección representativa** en lugar de la cola completa:
> QNodos corre 7 filas (n_max=12, 13×2, 17×3, 21) y Geometric corre 3 filas (n_max=12, 13, 17).
> Los datos faltantes de 20A Geometric y las filas 22A pendientes se documentan como
> limitación de tiempo de cómputo; no se infieren.

### Ejecución de los experimentos

Los scripts están en `scripts/` con symlinks en la raíz del proyecto para compatibilidad:

```
scripts/
  run_qnodos_single.py        # Una fila de 20A — QNodos
  run_qnodos_single_22A.py    # Una fila de 22A — QNodos
  run_qnodos_single_25A.py    # Una fila de 25A — QNodos
  run_geo_single.py           # Una fila de 20A — Geometric
  run_geo_single_22A.py       # Una fila de 22A — Geometric
  run_geo_single_25A.py       # Una fila de 25A — Geometric
  run_qnodos_cola.sh          # Cola completa 20A QNodos → 22A → 25A
  run_qnodos_cola_22A.sh      # Cola completa 22A QNodos → 25A
  run_qnodos_cola_25A.sh      # Cola completa 25A QNodos
  run_geo_cola.sh             # Cola completa 20A Geometric → 22A → 25A
  run_geo_cola_22A.sh         # Cola completa 22A Geometric → 25A
  run_geo_cola_25A.sh         # Cola completa 25A Geometric
```

Cada script de fila única acepta `--start-k` para reanudar desde una k intermedia. Las colas corren de a dos filas en paralelo y encadenan automáticamente el siguiente sistema al terminar.

```bash
# Ejecutar una fila específica
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -u scripts/run_qnodos_single.py 10

# Reanudar desde k=3
python3 -u scripts/run_geo_single_22A.py 45 --start-k 3

# Lanzar la cola completa (20A → 22A → 25A) en segundo plano
nohup bash run_qnodos_cola.sh > /tmp/qnodos_cola_master.log 2>&1 &
```

### Notas de rendimiento

- Coste por evaluación: O(2^mec) — mec=20 es 32× más caro que mec=15.
- Filas con mec=24 (QNodos k=3): ~37 min. Geometric k=3: ~16 h.
- Se usa `mmap_mode="r"` para compartir la TPM (~3 GB para N25A) entre procesos.
- Variables `OMP_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1` evitan la explosión de hilos de OpenBLAS.

### Optimizaciones aplicadas (sin pérdida de calidad)

| Módulo | Optimización | Impacto |
|--------|-------------|---------|
| `geometric.py` | DP hipercubo vectorizado con numpy (popcount-ordered) | ~100× menos overhead Python para n=15 |
| `geometric.py` | `_conductancias_geometrica()` vectorizada con `np.moveaxis` | 10–100× en Fiedler para n≥6 |
| `q_nodos.py` | SA termina si EMD=0 en estado inicial o por temperatura | Elimina ~1440 evaluaciones redundantes |
| `q_nodos.py` | Multi-start termina si EMD=0 (ahorra hasta 7 de 8 runs) | Hasta 8× menos trabajo en casos triviales |
| `k_particion_buscador.py` | `_recocido`, `_multi_recocido`, `refinar_local` con early exit EMD=0 | Aplica a k=3,4,5 en ambos algoritmos |
| `sistema.py` | `distribucion_marginal()` con `np.empty` (pre-alloc) | Reduce overhead en hot path (millones de llamadas) |
| `ncubo.py` | `marginalizar()` usa sets Python en vez de `np.intersect1d` | 11–23× más rápido para arrays pequeños (mec típico 10–25) |

---

## Muestras incluidas

`src/.samples/`: `N4A.csv`, `N5A.csv`, `N6A.csv`, `N7A.csv`, `N8A.csv`, `N20A.csv`, `N22A.csv`

Los archivos `.npy` de sistemas grandes (`N20A.npy`, `N22A.npy`, `N25A.npy`) están excluidos del repositorio por tamaño (`.gitignore`); se generan localmente con `scripts/gen_N25A.py` o cargando las muestras CSV.

---

## Flujo recomendado

```bash
source .venv/bin/activate
python exec.py                          # validación rápida
PYTHONPATH=. python -m pytest -q        # antes de cada commit
git add <archivos> && git commit -m "..."
git push origin main
```
