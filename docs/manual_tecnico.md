# Manual Técnico — Sistema MIP-IIT

**Proyecto:** Análisis de Irreducibilidad Sistémica (SIA) — Partición de Mínima Pérdida de Información  
**Repositorio:** ProyectoAnalisis2026  
**Autor:** CamiOso  
**Fecha:** Mayo 2026

---

## Tabla de contenidos

1. [Propósito del sistema](#1-propósito-del-sistema)
2. [Arquitectura general](#2-arquitectura-general)
3. [Capa de dominio](#3-capa-de-dominio)
4. [Capa de aplicación](#4-capa-de-aplicación)
5. [Capa de infraestructura](#5-capa-de-infraestructura)
6. [Estrategias](#6-estrategias)
7. [Flujo de datos extremo a extremo](#7-flujo-de-datos-extremo-a-extremo)
8. [Configuración](#8-configuración)
9. [Interfaz de línea de comandos](#9-interfaz-de-línea-de-comandos)
10. [Cómo agregar una nueva estrategia](#10-cómo-agregar-una-nueva-estrategia)
11. [Funciones utilitarias](#11-funciones-utilitarias)
12. [Pruebas](#12-pruebas)

---

## 1. Propósito del sistema

El sistema resuelve el problema de la **Partición de Mínima Información (MIP)** en el marco de la Teoría de la Información Integrada (IIT). Dado un sistema de `n` nodos con dinámica estocástica descrita por una **Matriz de Transición de Probabilidades (TPM)**, el objetivo es encontrar la bipartición (o k-partición) del sistema en partes que minimice la pérdida de información causada por el corte.

Formalmente, dada una TPM `T ∈ [0,1]^{2ⁿ × n}` y un estado inicial, se busca:

```
MIP = arg min_{(A,M)} EMD( dist_marginal_total, dist_marginal_partida(A, M) )
```

donde `A ⊆ alcance` y `M ⊆ mecanismo` definen el corte causal: se elimina la influencia de `M` sobre el complemento de `A` y del complemento de `M` sobre `A`.

El sistema implementa **14 estrategias** con enfoques que van desde la búsqueda exhaustiva hasta metaheurísticas y métodos de inspiración matemática avanzada.

---

## 2. Arquitectura general

El proyecto sigue la **Arquitectura Hexagonal** (Ports & Adapters, Alistair Cockburn). Las capas internas no dependen de las externas; las dependencias siempre apuntan hacia el centro.

```
┌────────────────────────────────────────────────────────────┐
│  PRESENTACIÓN          exec.py / src/main.py               │
│  (CLI, orquestador)                                        │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  APLICACIÓN             src/aplicacion/                     │
│  Casos de uso:          BuscarParticionOptima               │
│                         EstimarTPM                         │
│  Puertos (interfaces):  IEstrategia, IRegistro,            │
│                         IRepositorioTPM                    │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  DOMINIO                src/modelos/                        │
│  Entidades:             Sistema, NCube, Solucion           │
│  Enumeraciones:         MetricDistance, GeometricMode, ... │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  INFRAESTRUCTURA        src/infraestructura/ + adaptadores  │
│  Estrategias:           src/estrategias/*.py               │
│  Repositorio:           src/controladores/gestor.py        │
│  Logging:               src/intermedios/registro.py        │
│  Contenedor IoC:        src/contenedor.py                  │
└────────────────────────────────────────────────────────────┘
```

**Regla de dependencias:** únicamente `src/contenedor.py` importa de `src/infraestructura`. El resto del código depende de abstracciones (puertos).

### Estructura de directorios

```
ProyectoAnalisis2026/
├── exec.py                            # Punto de entrada CLI
├── src/
│   ├── main.py                        # Orquestación de la ejecución
│   ├── contenedor.py                  # Composition Root (IoC container)
│   ├── aplicacion/
│   │   ├── configuracion.py           # AppConfig (dataclass inmutable)
│   │   ├── casos_de_uso/
│   │   │   ├── buscar_particion.py    # BuscarParticionOptima
│   │   │   └── estimar_tpm.py         # EstimarTPM
│   │   └── puertos/
│   │       ├── estrategia.py          # IEstrategia (Protocol)
│   │       ├── registro.py            # IRegistro (Protocol)
│   │       └── repositorio_tpm.py     # IRepositorioTPM (Protocol)
│   ├── modelos/
│   │   ├── base/
│   │   │   └── sia.py                 # SIA (base abstracta de estrategias)
│   │   ├── nucleo/
│   │   │   ├── ncubo.py               # NCube
│   │   │   ├── sistema.py             # Sistema
│   │   │   └── solucion.py            # Solucion (DTO de salida)
│   │   └── enumeraciones/
│   │       ├── distancia.py           # MetricDistance
│   │       ├── geometric_mode.py      # GeometricMode
│   │       ├── emd_temporal.py        # TimeEMD
│   │       └── notacion.py            # Notation
│   ├── infraestructura/
│   │   ├── estrategias/__init__.py    # Re-exporta todas las estrategias
│   │   ├── repositorios/__init__.py   # Re-exporta Gestor
│   │   └── observabilidad/__init__.py # Re-exporta SafeLogger, perfilar
│   ├── estrategias/                   # Implementaciones concretas
│   │   ├── fuerza_bruta.py
│   │   ├── phi.py
│   │   ├── q_nodos.py
│   │   ├── geometrica.py
│   │   ├── circuito.py
│   │   ├── informacion_bottleneck.py
│   │   ├── louvain.py
│   │   ├── genetico.py
│   │   ├── particion_ilp.py
│   │   ├── belief_propagation.py
│   │   ├── remcmc.py
│   │   ├── variacional.py
│   │   ├── branch_bound.py
│   │   └── hiperbolica.py
│   ├── controladores/
│   │   └── gestor.py                  # Gestor (adaptador de repositorio)
│   ├── intermedios/
│   │   ├── registro.py                # SafeLogger
│   │   └── perfil.py                  # gestor_perfilado, decorador perfilar
│   ├── funciones/
│   │   ├── iit.py                     # EMD, distancias, conversiones
│   │   ├── formato.py                 # Formateo de salida
│   │   ├── particiones.py             # Generación de biparticiones
│   │   ├── grafo_info.py              # Grafo de acoplamientos
│   │   └── k_particion_buscador.py    # Infraestructura de k-partición
│   ├── constantes/
│   │   ├── models.py                  # Etiquetas de estrategias
│   │   ├── error.py                   # Mensajes de error
│   │   └── base.py                    # Constantes globales
│   └── .samples/                      # TPMs de muestra (CSV)
│       └── N{n}{pagina}.csv
├── tests/                             # Suite de pruebas
└── review/
    ├── notas/
    │   ├── bitacora_k_particiones.md  # Bitácora de diseño
    │   └── arquitectura.puml          # Diagrama PlantUML
    └── benchmarks/                    # Scripts de análisis comparativo
```

---

## 3. Capa de dominio

### 3.1 NCube (`src/modelos/nucleo/ncubo.py`)

Representa la distribución de probabilidad condicional de un nodo `i` como un tensor n-dimensional:

```python
@dataclass(frozen=True)
class NCube:
    indice: int              # Índice del nodo que este cubo representa
    dims:   NDArray[np.int8] # Índices de los nodos que influyen (dimensiones)
    data:   np.ndarray       # Tensor de forma (2,)*len(dims): P(X_i=1 | estado)
```

**Operaciones clave:**

| Método | Descripción |
|--------|-------------|
| `condicionar(indices, estado)` | Fija dimensiones de `indices` según `estado`; reduce el tensor |
| `marginalizar(ejes)` | Suma sobre los ejes dados; equivale a ignorar esas variables |
| `distribucion_marginal()` | Devuelve la distribución de probabilidad marginalizada a escalar |

### 3.2 Sistema (`src/modelos/nucleo/sistema.py`)

Colección de NCubes derivada de la TPM completa. Implementa las operaciones causales de la IIT:

```python
class Sistema:
    estado_inicial: NDArray[np.int8]  # Estado de inicio del sistema
    ncubos: tuple[NCube, ...]         # Un NCube por nodo
    memo:   dict[...]                 # Cache de subsistemas ya calculados
```

**Operaciones clave:**

| Método | Descripción |
|--------|-------------|
| `condicionar(indices)` | Aplica background conditions; elimina NCubes de esos índices |
| `substraer(alcance, mecanismo)` | Sustrae nodos del alcance (futuros) y mecanismo (dimensiones) |
| `bipartir(subalcance, submecanismo)` | Aplica el corte causal MIP: elimina aristas cruzadas |
| `k_bipartir(nodos, asignacion)` | Versión generalizada para k grupos |
| `distribucion_marginal()` | Distribución conjunta del subsistema |
| `indices_ncubos` | Propiedad: array de índices de NCubes activos |
| `dims_ncubos` | Propiedad: array de dimensiones activas |

**El corte causal `bipartir(A, M)`** elimina:
- La influencia de los nodos en `M` sobre los nodos en `complemento(A)`
- La influencia de los nodos en `complemento(M)` sobre los nodos en `A`

### 3.3 Solucion (`src/modelos/nucleo/solucion.py`)

DTO de salida que toda estrategia debe devolver:

```python
@dataclass
class Solucion:
    estrategia:               str          # Nombre de la estrategia
    perdida:                  float        # EMD mínimo encontrado (φ)
    distribucion_subsistema:  np.ndarray   # Distribución del sistema completo
    distribucion_particion:   np.ndarray   # Distribución del sistema partido
    estado_inicial:           str          # Estado inicial binario
    particion:                str          # Descripción textual de la bipartición
```

### 3.4 Enumeraciones

| Enumeración | Valores | Uso |
|-------------|---------|-----|
| `MetricDistance` | `HAMMING`, `WASSERSTEIN`, `JENSEN_SHANNON`, `FISHER_RAO`, `KL` | Métrica para calcular EMD |
| `GeometricMode` | `STRICT`, `REFINED` | Modo de búsqueda en Geometric |
| `TimeEMD` | `EMD_EFECTO`, `EMD_CAUSAL` | Dirección temporal del EMD |
| `Notation` | `LIL_ENDIAN`, `BIG_ENDIAN` | Orden de bits en los estados |

---

## 4. Capa de aplicación

### 4.1 Puertos (interfaces)

Los puertos son `Protocol` de Python — interfaces estructurales sin herencia:

**`IEstrategia` (`src/aplicacion/puertos/estrategia.py`)**
```python
class IEstrategia(Protocol):
    def aplicar_estrategia(
        self,
        estado_inicial: str,  # Cadena binaria, ej: "1010"
        condicion:      str,  # Binario: "1"=incluir, "0"=condicionar
        alcance:        str,  # Binario: "1"=incluir en alcance
        mecanismo:      str,  # Binario: "1"=incluir en mecanismo
        k:              int,  # Número de partes (default 2)
    ) -> Solucion: ...
```

**`IRegistro` (`src/aplicacion/puertos/registro.py`)**
```python
class IRegistro(Protocol):
    def debug(self, mensaje: str) -> None: ...
    def info(self, mensaje: str) -> None: ...
    def warn(self, mensaje: str) -> None: ...
    def error(self, mensaje: str) -> None: ...
```

**`IRepositorioTPM` (`src/aplicacion/puertos/repositorio_tpm.py`)**
```python
class IRepositorioTPM(Protocol):
    def cargar_red(self) -> np.ndarray: ...
    def construir_tpm_desde_csv_muestras(
        self, archivo: Path, valor_no_observado: float
    ) -> np.ndarray: ...
```

### 4.2 Casos de uso

**`BuscarParticionOptima` (`src/aplicacion/casos_de_uso/buscar_particion.py`)**

Orquesta la búsqueda delegando en la estrategia inyectada. No conoce qué estrategia concreta se usa.

```python
@dataclass
class EntradaBusqueda:
    estado_inicial: str
    condicion:      str
    alcance:        str
    mecanismo:      str
    k:              int = 2

@dataclass
class BuscarParticionOptima:
    estrategia: IEstrategia
    registro:   IRegistro

    def ejecutar(self, entrada: EntradaBusqueda) -> Solucion:
        self.registro.info(f"Iniciando búsqueda con {type(self.estrategia).__name__}")
        resultado = self.estrategia.aplicar_estrategia(
            entrada.estado_inicial,
            entrada.condicion,
            entrada.alcance,
            entrada.mecanismo,
            k=entrada.k,
        )
        self.registro.info(f"Partición encontrada: perdida={resultado.perdida:.4f}")
        return resultado
```

**`EstimarTPM` (`src/aplicacion/casos_de_uso/estimar_tpm.py`)**

Gestiona la obtención de la TPM desde cualquier fuente (CSV predefinido o muestras temporales).

### 4.3 Configuración (`src/aplicacion/configuracion.py`)

`AppConfig` es un `dataclass(frozen=True)` — inmutable e inyectable:

```python
@dataclass(frozen=True)
class AppConfig:
    semilla_numpy:     int = 73
    pagina_red_muestra: str = "A"
    distancia_metrica: str = MetricDistance.HAMMING.value
    notacion_indexado: str = Notation.LIL_ENDIAN.value
    tiempo_emd:        str = TimeEMD.EMD_EFECTO.value
    modo_geometrico:   str = GeometricMode.REFINED.value
```

---

## 5. Capa de infraestructura

### 5.1 Contenedor IoC (`src/contenedor.py`)

**Único punto** donde se instancian dependencias concretas. Ninguna otra capa importa de `src.infraestructura` directamente.

```python
class Contenedor:
    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()

    def estrategia(self, nombre: str, tpm: np.ndarray) -> IEstrategia:
        """Devuelve la estrategia concreta según el alias de nombre."""
        ...

    def caso_uso_buscar_particion(
        self, nombre_estrategia: str, tpm: np.ndarray
    ) -> BuscarParticionOptima:
        return BuscarParticionOptima(
            estrategia=self.estrategia(nombre_estrategia, tpm),
            registro=self.registro("busqueda"),
        )

    def caso_uso_estimar_tpm(self, estado_inicial: str) -> EstimarTPM:
        return EstimarTPM(
            repositorio=self.repositorio_tpm(estado_inicial),
            registro=self.registro("tpm"),
        )
```

**Aliases de estrategia registrados en el contenedor:**

| Alias(es) | Clase instanciada |
|-----------|-------------------|
| `fuerza_bruta`, `bruteforce`, `fuerzabruta` | `FuerzaBruta` |
| `phi` | `Phi` |
| `qnodos`, `q_nodes`, `qnodes` | `QNodos` |
| `geometric` | `Geometric` |
| `circuito` | `Circuito` |
| `ib`, `information_bottleneck`, `informacion_bottleneck` | `InformacionBottleneck` |
| `louvain` | `Louvain` |
| `genetico`, `ga`, `algoritmo_genetico` | `AlgoritmoGenetico` |
| `ilp`, `particion_ilp` | `ParticionILP` |
| `bp`, `belief_propagation` | `BeliefPropagation` |
| `remcmc`, `replica_exchange`, `parallel_tempering` | `REMCMC` |
| `variacional`, `particion_variacional` | `ParticionVariacional` (modo laplaciano) |
| `airy`, `biharmonico` | `ParticionVariacional` (modo biarmónico) |
| `bb`, `branch_bound`, `branchbound`, `branch_and_bound` | `BranchBound` |
| `hiperbolica`, `poincare`, `ryu_takayanagi` | `ParticionHiperbolica` |

### 5.2 Repositorio (`src/controladores/gestor.py`)

`Gestor` implementa `IRepositorioTPM` (duck typing):

```python
@dataclass
class Gestor:
    estado_inicial: str
    ruta_base: Path = Path("src/.samples")

    def cargar_red(self) -> np.ndarray:
        """Carga TPM predefinida desde src/.samples/N{n}{pagina}.csv."""

    def construir_tpm_desde_csv_muestras(
        self, archivo_muestras: Path, valor_no_observado: float = 0.5
    ) -> np.ndarray:
        """Estima TPM [2^n × n] desde secuencia temporal binaria.
        
        Para cada estado observado en t, acumula el estado t+1 por nodo.
        Estados no observados se asignan valor_no_observado (default 0.5).
        """
```

**Formato de los CSV de muestra:** tabla de `2^n` filas y `n` columnas, donde la fila `i` es la distribución de probabilidad condicional `P(X_{t+1} | estado=i)`. Sin encabezado; valores separados por comas.

**Formato de los CSV de muestras temporales:** filas = instantes de tiempo, columnas = nodos. Valores binarios (0 o 1).

### 5.3 Logging (`src/intermedios/registro.py`)

`SafeLogger` implementa `IRegistro`:

- Escribe simultáneamente a consola (nivel INFO) y a archivo `.logs/{nombre}.log` (nivel DEBUG).
- Evita duplicación de handlers si el logger ya fue inicializado.
- Thread-safe gracias al módulo `logging` de la biblioteca estándar.

### 5.4 Profiling (`src/intermedios/perfil.py`)

- `@perfilar`: decorador que mide el tiempo de ejecución de una función.
- `gestor_perfilado`: context manager para sessiones de profiling con `pyinstrument`.

---

## 6. Estrategias

Todas las estrategias heredan de `SIA` (`src/modelos/base/sia.py`), que provee:

- `sia_preparar_subsistema(estado, condicion, alcance, mecanismo)` — valida parámetros, crea el `Sistema`, aplica condicionamiento y sustracción, guarda `self.sia_subsistema` y `self.sia_dists_marginales`.
- `chequear_parametros(...)` — valida que los strings sean binarios y de longitud correcta.
- `seleccionar_emd(config)` — devuelve la función de distancia configurada.

### 6.1 FuerzaBruta

**Archivo:** `src/estrategias/fuerza_bruta.py`  
**Complejidad:** O(2^{n_a} · 2^{n_m})  
**Garantía:** exacta (óptimo global)

Enumera todas las biparticiones válidas del espacio `(subalcance × submecanismo)` y devuelve la de menor EMD. Es la referencia de correctitud para comparar con el resto de estrategias.

### 6.2 Phi

**Archivo:** `src/estrategias/phi.py`  
**Dependencia:** PyPhi (opcional)

Si PyPhi está instalado, usa su implementación de φ. Si no, cae a una heurística de respaldo compatible con el resto del sistema.

### 6.3 QNodos

**Archivo:** `src/estrategias/q_nodos.py`  
**Complejidad:** O(n²) para funciones submodulares; O(n²) + SA para las demás  
**Garantía:** exacta para funciones submodulares (~88% de sistemas aleatorios)

Implementa el algoritmo de Queyranne para minimización de funciones submodulares simétricas (MAO: Minimum Adjacent Order). Para el ~12% de funciones no submodulares, corre Simulated Annealing desde el resultado MAO como punto de inicio.

Internamente usa la infraestructura `BuscadorKParticion → BuscadorKRecocido` para k > 2.

### 6.4 Geometric

**Archivo:** `src/estrategias/geometrica.py` + `src/strategies/geometric.py`  
**Complejidad:** O(n·2^n) modo estricto; O(n·2^n) + hill-climbing modo refinado

Recorre el hipercubo binario de estados evaluando cortes en cada vértice. Dos modos:

- `STRICT`: tabla recursiva pura, cota teórica garantizada.
- `REFINED`: agrega restarts y hill-climbing para mayor precisión práctica.

Para k > 2 usa `BuscadorKDP` (programación dinámica de subconjuntos, O(3^n·k) + SA).

### 6.5 Circuito

**Archivo:** `src/estrategias/circuito.py`  
**Complejidad:** O(n³) (eigendescomposición)

Construye un grafo de acoplamientos donde el peso de la arista (i,j) es la sensibilidad de la TPM de `i` respecto al nodo `j`. Usa el **vector de Fiedler** (segundo eigenvector más pequeño del Laplaciano) para proponer la bipartición — la línea de menor corte espectral. Aplica refinamiento local de flip-de-un-nodo.

### 6.6 InformacionBottleneck

**Archivo:** `src/estrategias/informacion_bottleneck.py`  
**Complejidad:** O(n²·k·iter)

Agrupa nodos minimizando `I(nodos; partición)` sujeto a preservar `I(partición; efectos)`. Usa minimización alternada al estilo Tishby et al. (1999). Opera sobre el grafo de acoplamientos de la TPM.

### 6.7 Louvain

**Archivo:** `src/estrategias/louvain.py`  
**Complejidad:** O(n²·iter)

Detecta comunidades maximizando la modularidad `Q` del grafo de conductancias. Las comunidades corresponden a partes de la bipartición. Incluye fase de fusión de comunidades pequeñas para mapear al espacio (subalcance, submecanismo).

### 6.8 AlgoritmoGenetico

**Archivo:** `src/estrategias/genetico.py`  
**Complejidad:** O(generaciones · población · evaluación)

Metaheurística evolutiva: población de biparticiones codificadas como cromosomas binarios, selección por torneo, cruce uniforme, mutación bit-flip, elitismo. Incluye soporte para k > 2 con codificación entera.

### 6.9 ParticionILP

**Archivo:** `src/estrategias/particion_ilp.py`  
**Dependencia:** `scipy` (solver HiGHS vía `scipy.optimize.linprog`)

Formula el k-cut mínimo como Programa Lineal Entero y lo resuelve mediante relajación LP continua + redondeo. Para n pequeño puede ser exacto; para n grande es una cota inferior.

### 6.10 BeliefPropagation

**Archivo:** `src/estrategias/belief_propagation.py`  
**Complejidad:** O(iter · |aristas| · k²)

Aplica Loopy Belief Propagation sobre un Markov Random Field definido en el grafo de conductancias. Los mensajes sum-product convergen a asignaciones de nodos a partes. Incluye amortiguación (damping) para mejorar convergencia en grafos con ciclos.

### 6.11 REMCMC (Replica Exchange MCMC)

**Archivo:** `src/estrategias/remcmc.py`  
**Complejidad:** O(n_replicas · pasos)

Parallel Tempering con `n_replicas` cadenas Markov independientes a temperaturas escalonadas `T_1 < T_2 < … < T_r`. Periodicamente se proponen swaps entre cadenas adyacentes con probabilidad Metropolis-Hastings:

```
P(swap i,j) = min(1, exp((1/T_i - 1/T_j) · (φ_i - φ_j)))
```

La cadena fría (T=0) explota el óptimo actual; las calientes escapan de mínimos locales. Soporta k > 2.

### 6.12 ParticionVariacional

**Archivo:** `src/estrategias/variacional.py`  
**Complejidad:** O(n³) (eigendescomposición)

Opera sobre el grafo de conductancias de la TPM con dos modos:

**Modo `"laplaciano"`:** Usa el Laplaciano normalizado `L_n = D^{-½}·L·D^{-½}`. Minimiza el corte normalizado (Shi & Malik, 2000), favoreciendo particiones balanceadas en volumen. El vector de Fiedler de `L_n` define la bipartición.

**Modo `"biharmonico"` (Airy):** Usa el operador de Schrödinger `H = L + γ·diag(V)`, donde:
```
V[i] = 2·P(X_i = 1) - 1 = 2·mean(tpm[:, i]) - 1
```
`V[i] > 0` para nodos predominantemente activos; `V[i] < 0` para nodos inactivos. El turning point de Airy (donde `V=0`) marca la frontera natural de la partición. A diferencia del Laplaciano, `ev0` de `H` no es trivial y se incluye en el barrido de candidatos.

Ambos modos aplican un barrido de umbrales sobre eigenvectores sucesivos + refinamiento local.

### 6.13 BranchBound

**Archivo:** `src/estrategias/branch_bound.py`

Combina búsqueda exacta para sistemas pequeños con heurística híbrida para sistemas grandes:

**Fase exacta (`n_total ≤ umbral_exacto = 14`):**  
Enumera las `2^{n_a} · 2^{n_m} - 2` biparticiones válidas con cache. Garantiza el mismo resultado que `FuerzaBruta`.

**Fase heurística (`n_total > 14`):**
1. SA multi-arranque: `n_sa_arranques = 8` cadenas independientes con temperaturas iniciales escalonadas `T_i = 0.5·(1 + i·0.4)`. Aumenta la diversidad de exploración frente al SA de un único arranque de QNodos.
2. Expansión Hamming: desde el mejor SA, evalúa exhaustivamente todos los vecinos en la bola de Hamming `radio = 3` alrededor del mejor encontrado. Captura mínimos no submodulares próximos al óptimo.

**Con cache compartida** entre todas las evaluaciones de la misma sesión.

### 6.14 ParticionHiperbolica

**Archivo:** `src/estrategias/hiperbolica.py`

Inspirada en la **fórmula de Ryu-Takayanagi** (AdS/CFT, 2006): la entropía de entrelazamiento de una región equivale al área de la superficie geodésica mínima en el espacio Anti-de Sitter. El MIP-IIT es el análogo discreto: la "geodésica mínima" en el espacio hiperbólico inducido por la dinámica.

**Algoritmo:**

1. Construir conductancias `W` desde la TPM.
2. Eigenvectores 1 y 2 del Laplaciano → coordenadas `(x, y)`.
3. Proyección al disco de Poincaré: `coords_h = coords · tanh(2·‖coords‖) / ‖coords‖`.
4. Generar candidatos de dos familias:
   - **Geodésicas diametrales**: `n_angulos = 32` ángulos θ ∈ [0, π).
   - **Geodésicas circulares**: para cada par `(zᵢ, zⱼ)`, transformación de Möbius `T_{zᵢ}(z) = (z−zᵢ)/(1−z̄ᵢ·z)` que mapea `zᵢ → 0`. El signo de `Im[T(z_k)·conj(T(zⱼ)/|T(zⱼ)|)]` clasifica cada nodo.
5. Evaluar todos los candidatos y tomar el mínimo.
6. Refinamiento local flip-de-un-nodo.

Soporta k > 2 via embedding en `R^{k-1}` hiperbólico + k-means esférico.

---

## 7. Flujo de datos extremo a extremo

```
Usuario
  │
  │ python exec.py --estrategia qnodos --estado-inicial 1010
  ▼
exec.py
  │ args.estrategia = "qnodos"
  │ args.estado_inicial = "1010"
  ▼
src/main.py :: iniciar()
  │ tpm = contenedor.caso_uso_estimar_tpm("1010").cargar_desde_muestra_predefinida()
  │         └─→ Gestor.cargar_red() → src/.samples/N4A.csv → np.ndarray [16×4]
  │
  │ caso = contenedor.caso_uso_buscar_particion("qnodos", tpm)
  │         └─→ Contenedor.estrategia("qnodos", tpm) → QNodos(tpm, config)
  │         └─→ BuscarParticionOptima(estrategia=QNodos, registro=SafeLogger)
  │
  │ entrada = EntradaBusqueda(estado="1010", condicion="1111", alcance="1111", mecanismo="1111", k=2)
  │
  ▼
BuscarParticionOptima.ejecutar(entrada)
  │
  ▼
QNodos.aplicar_estrategia("1010", "1111", "1111", "1111", k=2)
  │
  ├─→ SIA.sia_preparar_subsistema(...)
  │     ├─ chequear_parametros: len=4, solo 0/1 ✓
  │     ├─ Sistema(tpm, estado=[1,0,1,0])
  │     ├─ sistema.condicionar([]) → sin cambio (condicion="1111")
  │     ├─ sistema.substraer(alcance_0=[], mec_0=[]) → sin cambio
  │     └─ self.sia_subsistema = sistema; self.sia_dists_marginales = distribucion_marginal()
  │
  ├─→ Algoritmo MAO Queyranne sobre (indices_ncubos, dims_ncubos)
  │     └─ Para cada bipartición candidata: sistema.bipartir(A, M).distribucion_marginal()
  │                                          └─→ EMD(dists_marginales, dist_partida)
  │
  └─→ Solucion(estrategia="QNodos", perdida=0.0436, particion="(M=(), A=(1,)) | ...")
        │
        ▼
  BuscarParticionOptima.ejecutar() retorna Solucion
        │
        ▼
  main.py imprime resultado
```

---

## 8. Configuración

### AppConfig

Todos los parámetros de comportamiento se centralizan en `AppConfig`. Se crea una vez en el `Contenedor` y se propaga a todas las estrategias sin usar singletons globales:

```python
from src.aplicacion.configuracion import AppConfig
from src.contenedor import Contenedor

# Configuración por defecto
contenedor = Contenedor()

# Configuración personalizada
config = AppConfig(
    distancia_metrica="wasserstein",
    modo_geometrico="estricto",
    pagina_red_muestra="B",
)
contenedor = Contenedor(config)
```

### Variables de AppConfig

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `semilla_numpy` | `int` | `73` | Semilla para reproducibilidad de RNG |
| `pagina_red_muestra` | `str` | `"A"` | Letra identificadora del CSV de muestra |
| `distancia_metrica` | `str` | `"hamming"` | Función EMD: `hamming`, `wasserstein`, `jensen_shannon`, `fisher_rao`, `kl` |
| `notacion_indexado` | `str` | `"lil_endian"` | Orden de bits: `lil_endian` o `big_endian` |
| `tiempo_emd` | `str` | `"emd_efecto"` | Dirección temporal: `emd_efecto` o `emd_causal` |
| `modo_geometrico` | `str` | `"refined"` | Modo de Geometric: `strict` o `refined` |

---

## 9. Interfaz de línea de comandos

```bash
python exec.py [opciones]
```

### Opciones principales

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--estrategia` | `todas` | Estrategia a ejecutar (ver alias en §5.1) |
| `--estado-inicial` | `1000` | Estado inicial binario (debe tener n bits) |
| `--modo-geometric` | `refinado` | Modo Geometric: `estricto` o `refinado` |
| `--k-particiones` | `2` | Número de partes de la partición |
| `--output-json` | `None` | Ruta para guardar resultado en JSON |
| `--csv-muestras` | `None` | Ruta a CSV de muestras temporales para estimar TPM |

### Ejemplos de uso

```bash
# Ejecutar todas las estrategias clásicas con el estado por defecto
python exec.py

# Usar QNodos con estado inicial 1010
python exec.py --estrategia qnodos --estado-inicial 1010

# Buscar bipartición hiperbólica con n=6 nodos
python exec.py --estrategia hiperbolica --estado-inicial 101010

# Búsqueda exacta con BranchBound
python exec.py --estrategia bb --estado-inicial 1000

# k=3 particiones con Geometric refinado
python exec.py --estrategia geometric --k-particiones 3 --estado-inicial 1000

# Guardar resultado en JSON
python exec.py --estrategia qnodos --output-json review/salidas/resultado.json

# Estimar TPM desde muestras temporales y luego buscar partición
python exec.py --estrategia circuito --csv-muestras datos/muestras.csv
```

### Uso programático

```python
from src.aplicacion.configuracion import AppConfig
from src.contenedor import Contenedor
from src.aplicacion.casos_de_uso.buscar_particion import EntradaBusqueda
import numpy as np

# Crear TPM para n=4 nodos
tpm = np.random.dirichlet(np.ones(4), size=16)  # [16 × 4]

# Configurar contenedor
config = AppConfig(distancia_metrica="wasserstein")
contenedor = Contenedor(config)

# Crear caso de uso
caso = contenedor.caso_uso_buscar_particion("qnodos", tpm)

# Ejecutar
entrada = EntradaBusqueda(
    estado_inicial="1010",
    condicion="1111",
    alcance="1111",
    mecanismo="1111",
    k=2,
)
resultado = caso.ejecutar(entrada)

print(resultado.perdida)   # φ mínima
print(resultado.particion) # "(M=(), A=(1,)) | (M*=(0,1,2,3), A*=(0,2,3))"
```

---

## 10. Cómo agregar una nueva estrategia

Agregar una estrategia requiere modificar **5 archivos** siguiendo el mismo patrón usado por las 14 estrategias existentes:

### Paso 1: Crear el archivo de la estrategia

```python
# src/estrategias/mi_estrategia.py
from __future__ import annotations
import numpy as np
from src.constantes.models import MI_LABEL
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class MiEstrategia(SIA):
    def __init__(self, tpm: np.ndarray, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solucion:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        alc_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mec_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        # === Tu algoritmo aquí ===
        mejor_alc, mejor_mec = alc_total[:1], ()
        mejor_perdida = float("inf")
        mejor_dist = self.sia_dists_marginales.copy()

        for subalc, submec in mis_candidatos(alc_total, mec_total):
            sp = self.sia_subsistema.bipartir(
                np.array(subalc, dtype=np.int8),
                np.array(submec, dtype=np.int8),
            )
            dist = sp.distribucion_marginal()
            perdida = float(self.distancia_metrica(self.sia_dists_marginales, dist))
            if perdida < mejor_perdida:
                mejor_perdida, mejor_dist = perdida, dist
                mejor_alc, mejor_mec = subalc, submec
        # ==========================

        return Solucion(
            estrategia=MI_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=fmt_biparticion(mejor_alc, mejor_mec, alc_total, mec_total),
        )
```

### Paso 2: Agregar la etiqueta de dominio

```python
# src/constantes/models.py
MI_LABEL: str = "MiEstrategia"
```

### Paso 3: Registrar en el Contenedor

```python
# src/contenedor.py — dentro del método estrategia()
if nombre_lower in {"mi_estrategia", "mie", "mi-alias"}:
    from src.estrategias.mi_estrategia import MiEstrategia
    return MiEstrategia(tpm, config=self._config)
```

También actualizar el mensaje de error al final del método para incluir el nuevo alias.

### Paso 4: Exportar desde la capa de infraestructura

```python
# src/infraestructura/estrategias/__init__.py
from src.estrategias.mi_estrategia import MiEstrategia

__all__ = [..., "MiEstrategia"]
```

### Paso 5: Verificar

```python
from src.aplicacion.configuracion import AppConfig
from src.contenedor import Contenedor

tpm = ...  # TPM de n nodos
caso = Contenedor(AppConfig()).caso_uso_buscar_particion("mi_estrategia", tpm)
from src.aplicacion.casos_de_uso.buscar_particion import EntradaBusqueda
n = tpm.shape[1]
resultado = caso.ejecutar(EntradaBusqueda("0"*n, "1"*n, "1"*n, "1"*n, k=2))
print(resultado)
```

---

## 11. Funciones utilitarias

### `src/funciones/iit.py`

| Función | Descripción |
|---------|-------------|
| `seleccionar_emd(config)` | Factory: devuelve la función de distancia según `config.distancia_metrica` |
| `emd_efecto(p, q)` | EMD L1 = Σ\|p−q\| normalizado |
| `emd_causal(p, q)` | Variante causal del EMD |
| `jensen_shannon(p, q)` | Divergencia JS = (KL(p\|m) + KL(q\|m))/2, m=(p+q)/2 |
| `wasserstein_sinkhorn(p, q)` | Distancia W1 via Sinkhorn-Knopp |
| `fisher_rao(p, q)` | Distancia geodésica de Fisher-Rao |
| `kl_divergencia(p, q)` | KL simétrica: (KL(p\|q) + KL(q\|p))/2 |
| `dec2bin(n, bits)` | Convierte entero a array binario de longitud `bits` |
| `estados_binarios(n)` | Genera todos los 2^n estados binarios |
| `big_endian(v)` / `lil_endian(v)` | Conversión de notación de bits |
| `generar_combinaciones(n)` | Genera todos los subconjuntos de {0,...,n-1} |

### `src/funciones/formato.py`

| Función | Descripción |
|---------|-------------|
| `fmt_biparticion(subalc, submec, alc_total, mec_total)` | Texto `"(M=..., A=...) \| (M*=..., A*=...)"` |
| `fmt_k_particion_asignacion(nodos, asig, alc, mec)` | Texto para k > 2 |
| `fmt_vector(values)` | Array numérico como `"[0.1234, 0.5678, ...]"` |
| `fmt_solution_block(...)` | Bloque completo de salida de una solución |

### `src/funciones/grafo_info.py`

| Función | Descripción |
|---------|-------------|
| `construir_afinidad(subsistema)` | Construye matriz de conductancias W[i,j] como sensibilidad cruzada de la TPM |

---

## 12. Pruebas

Las pruebas están en `tests/` y usan `pytest`:

```bash
# Ejecutar todas las pruebas
pytest tests/

# Prueba específica
pytest tests/test_strategy_q_nodes.py -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html
```

### Archivos de prueba

| Archivo | Cubre |
|---------|-------|
| `test_main.py` | Flujo completo de orquestación |
| `test_gestor.py` | Carga de TPMs, estimación desde CSV |
| `test_system.py` | `Sistema`, `NCube`, operaciones causales |
| `test_iit.py` | Funciones EMD y distancias |
| `test_cli.py` | Argumentos y salidas del CLI |
| `test_strategy_force.py` | FuerzaBruta como referencia |
| `test_strategy_phi.py` | Estrategia Phi |
| `test_strategy_geometric.py` | Geometric (estricto y refinado) |
| `test_strategy_q_nodes.py` | QNodos vs FuerzaBruta |
| `test_ejemplos_excel.py` | Casos de referencia del paper IIT |

### Benchmarks comparativos

En `review/benchmarks/` hay scripts para comparar estrategias en lotes:

```bash
# Benchmark de múltiples estrategias sobre sistemas aleatorios
python review/benchmarks/benchmark_comparativo.py

# Análisis de casos donde QNodos falla
python review/benchmarks/analisis_fallas_qnodos.py
```
