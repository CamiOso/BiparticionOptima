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

Todas las estrategias soportan **k-particiones** (k ≥ 2 grupos).

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
