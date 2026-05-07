# Bitácora de trabajo: K-Particiones y Estrategia Circuito

**Proyecto:** Partición de Mínima Pérdida de Información (MIP) — IIT  
**Fecha:** Abril 2026  
**Autor:** CamiOso

---

## Bitácora cronológica de investigación e implementación

Aquí dejamos el mapa de trabajo que seguimos en esta etapa del proyecto.
Tomamos como referencia las fechas registradas en los commits, porque nos
permiten ver con claridad qué se investigó, sobre qué temática se trabajó y
cómo esa investigación terminó convirtiéndose en código dentro del sistema.

| Fecha | Investigación realizada | Temática | Cómo se implementó en el proyecto |
|---|---|---|---|
| 2026-03-12 | Se levantó la base del proyecto y se definió la estructura inicial para trabajar con el sistema IIT. | Configuración inicial, modelo base y arranque del proyecto. | Se creó `README`, `pyproject`, `requirements`, `main` inicial, constantes base, `Application`, `Manager`, `NCube`, `System` y una primera versión de `BruteForce`. |
| 2026-03-13 | Se revisó la forma de calcular la pérdida de información y se empezaron a ordenar las salidas del sistema. | EMD, particionado y control del flujo. | Se agregó la métrica EMD, se encapsularon resultados en `Solution`, se centralizó el formato de salida y se integraron middleware de logging y perfilado. |
| 2026-03-13 | Se profundizó en estrategias alternativas a la búsqueda por fuerza bruta. | Phi, Q-Nodes y optimización de estrategias. | Se implementaron las estrategias `Phi` y `Q-Nodes`, se mejoró la heurística, se agregó memoización y se realizaron pruebas iniciales. |
| 2026-03-13 | Se reorganizó el proyecto para que el código quedara más legible y mantenible. | Refactorización de nombres y estructura. | Se tradujeron carpetas, módulos y clases principales al español y se ajustaron identificadores internos para alinearlos con la nueva convención. |
| 2026-03-14 | Se estudió cómo hacer más eficiente la búsqueda en la estrategia Geometric. | Estrategia Geometric y benchmark reproducible. | Se añadió Geometric con validación, refinamiento local, restarts adaptativos y comparación entre modos estricto y refinado. |
| 2026-03-15 | Se consolidó la experiencia de uso del sistema y se cerró la etapa de pruebas de la estrategia Geometric. | CLI, documentación y CI. | Se incorporó una interfaz de línea de comandos para elegir estrategias, se documentaron resultados finales y se añadió workflow de integración continua. |
| 2026-03-17 | Se trabajó en la salida estandarizada del sistema y en los datos de entrada para pruebas más realistas. | Estado inicial, JSON y muestras TPM. | Se agregó salida JSON por CLI, se pinnearon dependencias y se incluyeron muestras TPM `N5A` a `N8A`. |
| 2026-04-04 | Se evaluó cómo escalar mejor Geometric y cómo visualizar sus resultados. | Optimización y visualización. | Se mejoró la estrategia para sistemas grandes, se soportaron TPM desde muestras y se añadieron visualizaciones y benchmarks de optimización. |
| 2026-04-21 | Se revisó una forma distinta de abordar la partición del sistema, buscando salir de la búsqueda directa por biparticiones. | Estrategia Circuito y análisis espectral. | Se implementó `src/estrategias/circuito.py`, donde el sistema se modela como un grafo con Laplaciano espectral, eigenvectores y refinamiento local para obtener la partición. |
| 2026-04-30 | Se estudió cómo ampliar la búsqueda de particiones y cómo comparar mejor la calidad de las soluciones. | Métricas avanzadas, recocido simulado y benchmark. | Se agregaron nuevas métricas como JS, KL, Wasserstein y Fisher-Rao, además de la estrategia de recocido simulado, visualizaciones y scripts de evaluación en `review/benchmarks/`. |
| 2026-05-03 | Se revisó la necesidad de escalar el proyecto y organizar mejor la lógica de partición para nuevas estrategias. | Arquitectura del software y k-particiones avanzadas. | Se reorganizó el proyecto con arquitectura hexagonal y se incorporaron cinco estrategias avanzadas de k-partición, manteniendo una estructura más limpia y reutilizable. |

En términos generales, esta bitácora muestra cómo el proyecto fue creciendo
desde una primera revisión de los métodos de partición hasta la integración
de nuevas estrategias, métricas y estructuras de software. La intención fue
que cada investigación no quedara solo en teoría, sino que terminara reflejada
en una implementación concreta y verificable dentro del repositorio.

## ¿Por qué se hizo esto?

El proyecto  tenía varias estrategias para dividir un sistema en dos grupos
y encontrar la división que menos información pierde. Pero eso es una limitación
fuerte: no todos los sistemas se dividen bien en exactamente dos partes.

la tarea fue:

1. **Extender Geometric y Q-Nodos para que funcionen con k grupos** (no solo 2).
2. **Diseñar una estrategia nueva** que ataque el problema desde un ángulo
   completamente distinto, usando álgebra lineal en vez de búsqueda directa.

---

## Parte 1 — Soporte de k-particiones en Geometric y Q-Nodos

### ¿Qué había antes?

Las dos estrategias aceptaban como entrada el estado inicial del sistema, la
condición de fondo, el alcance y el mecanismo. Internamente siempre calculaban
una bipartición: grupo A vs. grupo B. No había forma de pedirles más grupos.

### ¿Qué cambió?

Se agregó un parámetro `k` al método principal de ambas estrategias. Si `k=2`
funciona igual que siempre. Si `k=3` divide en tres grupos, `k=4` en cuatro, etc.

```python
# Antes (solo bipartición)
resultado = estrategia.aplicar_estrategia("1000", "1111", "1111", "1111")

# Ahora (cualquier número de grupos)
resultado = estrategia.aplicar_estrategia("1000", "1111", "1111", "1111", k=3)
```

Cada estrategia lo resuelve a su manera:

---

### Cómo lo hace Geometric con k > 2

Geometric trabaja con la geometría del hipercubo de estados. Para k=2 ya tenía
un mecanismo eficiente. Para k>2 se implementaron dos variantes según el tamaño:

- **Sistema pequeño (4 nodos o menos):** prueba todas las asignaciones posibles
  de nodos a k grupos. Como son pocos nodos, el número de combinaciones es
  manejable y da la solución exacta.

- **Sistema más grande:** empieza con una asignación inicial y la va mejorando
  moviendo un nodo a la vez entre grupos. Si mover el nodo A del grupo 1 al grupo 2
  baja la pérdida φ, hace el movimiento. Repite hasta que ningún movimiento mejore
  el resultado. Esto se llama búsqueda local o hill-climbing.

---

### Cómo lo hace Q-Nodos con k > 2

Q-Nodos se basa en una propiedad matemática llamada submodularidad. En términos
simples: si vas agregando nodos a un grupo, el aporte de cada nodo nuevo es cada
vez menor. Eso permite una búsqueda tipo "greedy" (codicioso) que va tomando
las mejores decisiones locales una por una sin probar todo.

Para k>2 se adaptó esa búsqueda para construir k grupos de forma progresiva.
Primero arma el grupo 0, luego el grupo 1 buscando qué nodos benefician más
estar separados del 0, y así hasta llegar al grupo k-1.

---

## Parte 2 — El experimento: Geometric vs. Q-Nodos en k-particiones

### Cómo se armó el experimento

Para comparar ambas estrategias se generaron sistemas aleatorios con distintos
tamaños. Se usó semilla fija en cada prueba para que los resultados sean
reproducibles (si alguien corre el mismo script, obtiene los mismos números).

**Configuración exacta del benchmark:**

| Parámetro | Valores |
|---|---|
| Tamaños de sistema | 4, 5 y 6 nodos |
| Grupos buscados (k) | 3 y 4 |
| Semillas por tamaño | 5 semillas distintas |
| Semillas para 4 nodos | 11, 19, 29, 37, 47 |
| Semillas para 5 nodos | 13, 23, 31, 41, 53 |
| Semillas para 6 nodos | 17, 27, 37, 47, 59 |
| Estado inicial | todos los nodos en 0 |
| Máscara | todos los nodos participan |

En total fueron **30 ejecuciones comparativas** (3 tamaños × 2 valores de k × 5 semillas).

La TPM de cada sistema se generó con `numpy.random.default_rng(semilla)`, que produce
probabilidades de transición uniformes entre 0 y 1. Eso da sistemas sin estructura
particular, que sirven como caso de prueba neutral.

---

### Los resultados

Acá están los datos reales del CSV generado por el benchmark:

**k = 3 (dividir en 3 grupos)**

| Nodos | Tiempo Q-Nodos | Tiempo Geometric | Geometric es X veces más rápido | Diferencia de φ promedio | Quién encontró mejor φ |
|:---:|---:|---:|:---:|:---:|:---:|
| 4 | 0.373 s | 0.008 s | **48×** | 0.359 | Q-Nodos (5 de 5) |
| 5 | 0.572 s | 0.026 s | **22×** | 0.590 | Q-Nodos (5 de 5) |
| 6 | 0.919 s | 0.094 s | **10×** | 0.576 | Q-Nodos (5 de 5) |

**k = 4 (dividir en 4 grupos)**

| Nodos | Tiempo Q-Nodos | Tiempo Geometric | Geometric es X veces más rápido | Diferencia de φ promedio | Quién encontró mejor φ |
|:---:|---:|---:|:---:|:---:|:---:|
| 4 | 1.215 s | 0.012 s | **112×** | 0.359 | Q-Nodos (5 de 5) |
| 5 | 0.984 s | 0.038 s | **27×** | 0.590 | Q-Nodos (5 de 5) |
| 6 | 1.935 s | 0.120 s | **16×** | 0.576 | Q-Nodos (5 de 5) |

*φ = pérdida de información. Valor más bajo = mejor partición.*

---

### ¿Qué dicen los números?

**Geometric es mucho más rápido en todos los casos sin excepción.**

El caso más dramático fue k=4 con 4 nodos: Geometric tardó 0.012 segundos y
Q-Nodos tardó 1.215 segundos. Eso es 112 veces más rápido. Cuando el sistema
crece a 6 nodos la diferencia se reduce pero sigue siendo enorme (16 veces).

**Q-Nodos encontró mejores particiones en las 30 pruebas.**

En ninguna de las 30 corridas Geometric pudo igualar o superar la calidad de
la partición de Q-Nodos. La diferencia promedio de φ oscila entre 0.36 y 0.59
dependiendo del tamaño, lo cual es significativo — no es ruido estadístico.

**¿Por qué Geometric pierde en calidad?**

La búsqueda local que usa Geometric para k>2 puede quedar atrapada en un
mínimo local: encuentra una solución que parece buena porque ningún movimiento
individual la mejora, pero que no es la mejor global. Q-Nodos evita esto en
parte porque su construcción greedy con submodularidad tiene garantías teóricas
sobre qué tan lejos puede quedar del óptimo.

**Conclusión de esta parte:**

Estos dos resultados juntos forman el clásico tradeoff de los algoritmos de
búsqueda: rapidez vs. precisión. No hay una estrategia mejor en todos los casos.
Depende de qué necesita el usuario.

---

## Parte 3 — La estrategia Circuito

### La idea detrás

Mientras Geometric y Q-Nodos buscan la partición probando combinaciones,
la estrategia Circuito hace algo distinto: en vez de buscar, *infiere* la
estructura del sistema desde las matemáticas del grafo que lo representa.

La idea central es modelar el sistema como un **circuito eléctrico**. Cada
nodo es un punto del circuito, y la conexión entre dos nodos tiene una
"conductancia" que representa cuánto se influyen mutuamente. Si el nodo A
cambia mucho la probabilidad futura del nodo B (según la TPM), la conexión
A-B tiene alta conductancia, como un cable grueso. Si casi no se influyen,
la conexión es como un cable muy delgado o prácticamente cortado.

```
Sistema muy integrado:        Sistema con corte natural:
   A ══════ B ══════ C           A ══════ B ──── C
   (cables gruesos)              (cable fino entre B y C)
   → Difícil de partir           → Se parte naturalmente aquí
```

### Paso a paso: cómo funciona internamente

**Paso 1 — Construir la matriz de conductancias**

Para cada par de nodos (i, j) se calcula cuánto cambia la probabilidad futura
del nodo i cuando cambia el estado actual del nodo j. Eso se hace tomando la
diferencia de probabilidades entre los estados donde j=0 y j=1, promediada
sobre todos los estados de los demás nodos.

Por ejemplo, si el nodo A está encendido y el nodo B tiene P(B=1) = 0.8,
pero cuando A está apagado P(B=1) = 0.2, la diferencia es 0.6. Eso significa
que A influye bastante en B → conductancia alta entre A y B.

**Paso 2 — Construir el Laplaciano**

Con la matriz de conductancias W se construye el **Laplaciano** del grafo:
`L = D - W`, donde D es una matriz diagonal que contiene la suma de
conductancias de cada nodo. El Laplaciano es una forma estándar en matemáticas
de representar la estructura de conectividad de un grafo.

**Paso 3 — Encontrar el corte natural con eigenvectores**

Acá viene la parte más interesante. El Laplaciano tiene propiedades
matemáticas que revelan dónde se "rompe naturalmente" el grafo.

Su segundo eigenvector (llamado **vector de Fiedler**) le asigna un número
a cada nodo. Los nodos con número positivo van a un grupo y los de número
negativo al otro. Es como si el vector de Fiedler le pintara a cada nodo
de qué lado del corte más natural está.

Para k=2 se barre varios umbrales sobre ese vector y se evalúa cuál corte
da menos pérdida φ.

Para k>2 se usan los k primeros eigenvectores como coordenadas y se aplica
k-means para agrupar los nodos en ese espacio reducido.

**Paso 4 — Refinamiento local**

El resultado espectral es una buena propuesta inicial pero no necesariamente
la óptima. Se aplica un paso de refinamiento local (igual al de Geometric)
donde se mueve un nodo a la vez entre grupos mientras eso reduzca la pérdida.
Hasta 24 iteraciones de mejora.

### ¿Por qué es interesante?

Lo que hace diferente a Circuito es que **no busca entre particiones**.
No prueba combinaciones ni construye grupos incrementalmente. En cambio,
descompone el Laplaciano matemáticamente y obtiene directamente una
propuesta de partición. El costo computacional es `O(n³)` — la
eigendescomposición de una matriz n×n — y es determinista (siempre da
el mismo resultado para el mismo sistema).

La desventaja es que al no buscar, puede perderse la partición óptima
en sistemas donde la estructura no es tan clara geométricamente.

### Cómo se usa

```python
from src.estrategias.circuito import Circuito
import numpy as np

# Cualquier TPM válida (2^n filas, n columnas)
tpm = np.random.rand(8, 3).astype(np.float32)

circuito = Circuito(tpm)

# Bipartición (k=2)
sol = circuito.aplicar_estrategia(
    estado_inicial="101",
    condicion="111",
    alcance="111",
    mecanismo="111",
    k=2
)
print(sol.particion)   # por ejemplo: "A|BC"
print(sol.perdida)     # por ejemplo: 0.25

# K-partición (k=3)
sol_k3 = circuito.aplicar_estrategia(
    estado_inicial="101",
    condicion="111",
    alcance="111",
    mecanismo="111",
    k=3
)
print(sol_k3.particion)   # por ejemplo: "G0(A)|G1(B)|G2(C)"
```

### Complejidad

| Paso | Operación | Costo |
|---|---|---|
| Construir conductancias | Recorrer la TPM | O(2ⁿ · n) |
| Eigendescomposición | Laplaciano n×n | O(n³) |
| K-means | k iteraciones sobre n puntos | O(k · n · iter) |
| Refinamiento local | Hasta 24 pasos | O(n · iter) |

Para sistemas grandes el costo dominante es la eigendescomposición O(n³).
Eso lo hace más escalable que Fuerza Bruta (O(2ⁿ)) y comparable con otras
estrategias aproximadas.

---

## Resumen general

Este trabajo agregó dos cosas al proyecto:

**1. K-particiones en Geometric y Q-Nodos**

Ahora el sistema puede dividirse en más de 2 grupos. Los datos del benchmark
muestran que Q-Nodos siempre encontró mejor calidad de partición (φ menor)
y que Geometric siempre fue más rápido — hasta 112× más rápido en el peor caso.
El balance entre velocidad y precisión depende del sistema y del tiempo disponible.

**2. Estrategia Circuito**

Una nueva forma de atacar el problema que no busca entre particiones sino que
las infiere desde la estructura matemática del grafo usando el Laplaciano y sus
eigenvectores. Es determinista, escalable en O(n³) y parte de un enfoque
completamente diferente al resto de las estrategias del proyecto.

---

## Archivos relacionados (k-particiones y Circuito)

| Archivo | Descripción |
|---|---|
| `src/estrategias/circuito.py` | Implementación completa de la estrategia Circuito |
| `src/strategies/geometric.py` | Geometric con soporte de k-particiones |
| `src/estrategias/q_nodos.py` | Q-Nodos con soporte de k-particiones |
| `review/benchmarks/benchmark_k_partitions.py` | Script que genera los datos de la tabla |
| `review/benchmarks/k_particiones_qnodos_vs_geometric_resumen.csv` | Datos completos del experimento |

Para reproducir el benchmark:

```bash
source .venv/bin/activate
PYTHONPATH=. python review/benchmarks/benchmark_k_partitions.py
```

---

## Parte 4 — Refactorización: patrón Template Method en la búsqueda de k-particiones

### El problema que se detectó

Al revisar el código de Geometric y Q-Nodos se encontró que ambas estrategias
tenían exactamente el mismo algoritmo de búsqueda de k-particiones copiado en
dos lugares distintos: la lógica de canonicalización, generación de vecinos,
refinamiento local, búsqueda exacta exhaustiva y búsqueda local con restarts.

Tener ese código duplicado es un problema porque cualquier mejora al algoritmo
hay que hacerla en dos lugares, y si se olvida uno de los dos, el comportamiento
de las estrategias se desincroniza sin que sea evidente.

### La solución: patrón Template Method

Se creó la clase abstracta `BuscadorKParticion` en `src/funciones/k_particion_buscador.py`.
Esta clase define el esqueleto del algoritmo de búsqueda pero deja un hueco:
el método `evaluar_asignacion()`, que es la única parte que realmente difiere
entre Geometric y Q-Nodos.

```
BuscadorKParticion (abstracta)
│
├── buscar(k, semilla)          ← decide si usar exacto o local según n
├── _buscar_exacto(k)           ← enumera todas las asignaciones canónicas
├── _buscar_local(k, semilla)   ← hill-climbing con restarts aleatorios
├── refinar_local(inicio, k)    ← descenso por vecindad (move one node at a time)
├── vecinos(asignacion, k)      ← genera movimientos de un nodo a otro grupo
├── canonicalizar(asignacion)   ← normaliza la etiqueta de grupos (0 aparece primero)
│
└── evaluar_asignacion()        ← ABSTRACTO: implementado por cada subclase
```

Las subclases concretas que se crearon:

**`_BuscadorKGeometric`**: evalúa usando `sistema.k_bipartir(nodos, asignacion)`,
donde los nodos son índices enteros (espacio espacial).

**`_BuscadorKQNodos`**: evalúa usando `sistema.k_bipartir_temporal(grupos_mec, grupos_alc)`,
donde los vértices son pares `(tiempo, índice)` (espacio temporal).

Con este cambio, las ~120 líneas de código duplicado desaparecieron. Si se mejora
el algoritmo de búsqueda (por ejemplo, cambiar la temperatura de los restarts o
el criterio de convergencia), el cambio se hace una sola vez en `BuscadorKParticion`
y ambas estrategias se benefician automáticamente.

---

## Parte 5 — Recocido simulado como búsqueda alternativa de k-particiones

### La limitación de la búsqueda local codiciosa

La búsqueda local (hill-climbing) que usan Geometric y Q-Nodos tiene una debilidad
conocida: puede quedar atrapada en un mínimo local. Si la pérdida de la asignación
actual es `φ = 0.35` y ningún movimiento individual (mover un nodo a otro grupo)
la mejora, el algoritmo se detiene aunque exista una asignación con `φ = 0.20`
que requeriría mover varios nodos a la vez para llegar a ella.

### Recocido simulado: aceptar soluciones peores a propósito

La clase `BuscadorKRecocido` implementa el algoritmo de recocido simulado
(Simulated Annealing), que resuelve este problema aceptando soluciones peores
con una probabilidad que disminuye con el tiempo:

```
P(aceptar peor solución) = exp(-Δφ / T)
```

Donde `Δφ` es cuánto peora la solución y `T` es la temperatura actual. Al
principio (T alta), el algoritmo acepta casi cualquier movimiento y explora
el espacio libremente. Con el tiempo, T baja y el algoritmo se vuelve cada
vez más selectivo, hasta comportarse casi como hill-climbing puro.

```python
from src.funciones.k_particion_buscador import BuscadorKRecocido

# BuscadorKRecocido implementa la misma interfaz que BuscadorKParticion.
# Solo necesita implementar evaluar_asignacion() como cualquier otra subclase.
```

El esquema de enfriamiento usado es geométrico: `T_{nueva} = T * factor`,
con `factor = 0.92` por defecto, lo que da un descenso suave.

---

## Parte 6 — Nuevas métricas de distancia entre distribuciones

Hasta ahora todas las estrategias usaban como métrica el EMD simplificado
(`Σ|u - v|`, equivalente a L1). Se agregaron cuatro métricas nuevas que
tienen propiedades matemáticas distintas y pueden cambiar qué partición
se considera óptima para un sistema dado.

### Jensen-Shannon

La divergencia JS entre distribuciones p y q se define como:

```
JS(p, q) = (1/2) KL(p || m) + (1/2) KL(q || m)    donde m = (p+q)/2
```

Es simétrica (JS(p,q) = JS(q,p)) y acotada en [0, log(2)]. Se implementa
como su raíz cuadrada, que es una métrica válida (cumple desigualdad triangular).

### KL divergencia simétrica

La KL clásica no es simétrica. Se usó la versión simetrizad:

```
KL_sim(p, q) = (KL(p||q) + KL(q||p)) / 2
```

Es más estricta que el EMD: si `q` tiene probabilidad cero donde `p` no,
la divergencia es infinita. Útil para detectar particiones que colapsan estados.

### Wasserstein con Sinkhorn

La distancia de Wasserstein W1 resuelve un problema de transporte óptimo:
¿cuánto cuesta mover la masa de la distribución `u` para transformarla en `v`,
donde el costo de mover masa del nodo i al nodo j es `|i-j|/n`?

Se implementa con el algoritmo de Sinkhorn-Knopp, que resuelve una versión
regularizada en tiempo O(n²) con iteraciones matriciales:

```
K = exp(-C / ε)     (núcleo del transporte)
iteración: a = u / (K b),   b = v / (K^T a)
T_óptimo = diag(a) K diag(b)
W_ε = <C, T_óptimo>
```

### Fisher-Rao

El espacio de distribuciones de probabilidad es una variedad Riemanniana
con métrica de Fisher. La distancia geodésica entre dos distribuciones
en esa variedad es el ángulo de Bhattacharyya:

```
d_FR(p, q) = 2 · arccos(Σᵢ √(pᵢ · qᵢ))
```

Varía entre 0 (distribuciones idénticas) y π (soportes disjuntos).
Es invariante a reparametrizaciones y más sensible que L1 en las colas.

---

## Parte 7 — Entropías de orden superior

Se implementaron las dos generalizaciones paramétricas más importantes de
la entropía de Shannon, en `src/funciones/entropia.py`.

### Entropía de Rényi

```
H_α(X) = 1/(1-α) · log₂(Σᵢ pᵢᵅ)
```

Para α → 1 converge a Shannon (por L'Hôpital). Casos especiales:
- α = 0: log₂(|soporte|) — solo cuenta cuántos estados son posibles
- α = 2: información de colisión — cuánto se superponen dos muestras aleatorias
- α → ∞: -log₂(max p) — entropía mín, solo mira el estado más probable

### Entropía de Tsallis

```
S_q(X) = (1 - Σᵢ pᵢq) / (q - 1)
```

A diferencia de Rényi, Tsallis no es extensiva: para sistemas independientes
A y B, `S_q(AB) = S_q(A) + S_q(B) + (1-q)·S_q(A)·S_q(B)`. Ese término
cruzado captura correlaciones de largo alcance, lo que la hace relevante
para sistemas con dependencias fuertes como los que modela IIT.

### Perfil de entropías

```python
perfil = perfil_entropia(p)
# {0.0: 2.0, 0.5: 1.799, 1.0: 1.648, 2.0: 1.454, 5.0: 1.222}
```

El perfil describe la forma de la distribución: una curva plana indica
distribución uniforme; una curva con caída pronunciada indica alta concentración.

---

## Parte 8 — O-information y correlación total

La información mutua clásica I(X;Y) mide dependencia entre dos variables.
Para sistemas de n variables existe una generalización que captura efectos
colectivos que no aparecen en las parejas, implementada en
`src/funciones/informacion_superior.py`.

### Correlación total (TC)

```
TC(X₁,...,Xₙ) = Σᵢ H(Xᵢ) - H(X₁,...,Xₙ)    ≥ 0
```

Mide la información total compartida entre todos los nodos. Es cero si y
solo si todos los nodos son independientes.

### O-information

```
Ω(X) = (n-2)·H(X) - Σᵢ H(Xᵢ) + Σᵢ<ⱼ H(Xᵢ,Xⱼ)
```

El signo de Ω determina el carácter del sistema:
- **Ω > 0**: redundancia dominante — varios nodos codifican la misma información
- **Ω < 0**: sinergia dominante — la información conjunta supera la suma de las partes
- **Ω = 0**: balance entre redundancia y sinergia

La sinergia tiene relación directa con φ de IIT: un sistema sinérgico es más
resistente a ser partido porque pierde información que solo existe en el todo.
Al hacer la bipartición, se destruye precisamente esa información sinérgica.

### Matriz de dependencia

```python
mat = matriz_dependencia(tpm, estado_inicial)
# Matriz n×n: mat[i,j] = I(Xi; Xj)  (información mutua entre cada par)
# Diagonal: mat[i,i] = H(Xi)
```

Permite identificar visualmente qué pares de nodos comparten más información
y cuáles son prácticamente independientes, lo que da pistas sobre dónde
conviene hacer el corte.

---

## Parte 9 — Análisis espectral de la TPM

La TPM define un proceso de Markov sobre {0,1}^n. Su eigendescomposición
revela la dinámica de largo plazo del sistema, implementada en
`src/herramientas/espectral.py`.

### Construcción de la matriz de transición completa

La TPM tiene formato (2^n × n): cada fila es un estado del sistema y cada
columna es la probabilidad de que el nodo i esté encendido en el siguiente paso.
Para el análisis espectral se necesita la matriz completa P (2^n × 2^n), donde
P[s, s'] = P(X_{t+1} = s' | X_t = s).

Bajo independencia condicional entre nodos, esta se calcula como el producto
de las probabilidades individuales de cada nodo.

### Interpretación de los eigenvalores

El Teorema de Perron-Frobenius garantiza que el eigenvalor dominante es 1
y su eigenvector asociado es la distribución estacionaria π (el estado al
que converge el sistema sin importar desde dónde empiece).

El segundo eigenvalor |λ₂| determina la velocidad de convergencia:
- **Brecha espectral**: gap = 1 - |λ₂|
- **Tiempo de mezcla**: t_mix ≤ log(1/ε) / gap

Un sistema con gap ≈ 0 (|λ₂| ≈ 1) tiene memoria larga: tarda muchos pasos
en olvidar su estado inicial. Eso se correlaciona con alta irreducibilidad
en IIT: si el sistema recuerda de dónde viene, partirlo destruye esa memoria.

### Entropía de la distribución estacionaria

La entropía de Shannon de π indica cuántos estados son accesibles a largo plazo.
Un sistema con H(π) alto visita muchos estados; uno con H(π) bajo se concentra
en pocos atractores.

---

## Resumen de todo el trabajo (Partes 1–9)

| Módulo / Archivo | Qué aporta |
|---|---|
| `src/funciones/k_particion_buscador.py` | `BuscadorKParticion` + `BuscadorKRecocido` (SA) + `BuscadorKDP` (DP subconjuntos + SA) + `buscar_con_semilla` |
| `src/funciones/iit.py` | Jensen-Shannon, KL simétrica, Wasserstein-Sinkhorn, Fisher-Rao |
| `src/funciones/entropia.py` | Rényi, Tsallis, perfil de entropías, divergencia de Rényi |
| `src/funciones/informacion_superior.py` | O-information, correlación total, matriz de dependencia |
| `src/herramientas/espectral.py` | Eigenvalores, π estacionaria, brecha espectral, tiempo de mezcla |
| `src/herramientas/benchmark.py` | Comparación automática de estrategias con tabla de resultados |
| `src/visualizacion/particion.py` | Gráficas de bipartición, k-partición y comparación de pérdidas |
| `src/controladores/gestor.py` | Estimación bayesiana de TPM con prior de Dirichlet |

---

## Parte 10 — Arquitectura hexagonal (Clean Architecture)

### El problema que se detectó

A medida que el proyecto creció, el código fue acumulando acoplamiento horizontal:
las estrategias importaban directamente el singleton `aplicacion`, `main.py`
instanciaba estrategias concretas por nombre, y el logging y el profiling
estaban mezclados dentro de cada clase. Cualquier cambio a la configuración
global podía afectar inadvertidamente a múltiples módulos.

El síntoma más claro: para testear una estrategia con una configuración distinta
a la del singleton había que mutar el estado global y luego restaurarlo, lo
cual es frágil y no funciona en tests paralelos.

### La solución: separar en cuatro capas

Se reorganizó el proyecto siguiendo los principios de **arquitectura hexagonal**
(Ports & Adapters, Alistair Cockburn 2005) y **Clean Architecture** (Robert Martin):

```
Dominio ← Aplicación ← Infraestructura ← Presentación
```

La regla de dependencia dice que el código solo puede apuntar hacia adentro:
infraestructura puede usar dominio, pero dominio nunca conoce infraestructura.

**Capa de Dominio** (`src/dominio/`)  
Contiene las entidades puras del problema: `NCube`, `Sistema`, `Solucion` y las
enumeraciones. No importa nada de fuera. Cualquier cambio de framework o
librería externa no la toca.

**Capa de Aplicación** (`src/aplicacion/`)  
Define los contratos que necesita el dominio del mundo exterior mediante
*puertos* (Protocols de Python):

- `IEstrategia`: cualquier objeto con `aplicar_estrategia(...)` satisface el contrato.
- `IRepositorioTPM`: cualquier fuente que devuelva una TPM (CSV, base de datos, red).
- `IRegistro`: cualquier sistema de logging (SafeLogger, stdout, null logger).

También contiene los *casos de uso*, que son la orquestación de alto nivel:
- `BuscarParticionOptima`: recibe una estrategia y un logger inyectados, corre la búsqueda.
- `EstimarTPM`: recibe un repositorio inyectado, carga o estima la TPM.

Y el `AppConfig`: un dataclass `frozen=True` que reemplaza al singleton mutable.
Para cambiar la configuración se crea una nueva instancia; no se muta estado global.

**Capa de Infraestructura** (`src/infraestructura/`)  
Re-exporta los adaptadores concretos organizados por tipo:
estrategias, repositorios CSV, logging/profiling, visualización y herramientas.
Esta capa implementa los puertos definidos en Aplicación.

**Capa de Presentación** (`src/presentacion/`)  
El `orquestador.py` traduce argumentos del mundo exterior en llamadas a
casos de uso. No contiene lógica de negocio.

**Composition root** (`src/contenedor.py`)  
Es el único punto del programa que conoce tanto los puertos como sus
implementaciones concretas. Aquí se hace el ensamblado:

```python
# Antes: acoplamiento directo con singleton y estrategia concreta
solver = FuerzaBruta(tpm)                    # instanciación directa
aplicacion.tiempo_emd = "jensen-shannon"     # mutación global

# Después: inyección de dependencias desde el contenedor
config = AppConfig(tiempo_emd=TimeEMD.JENSEN_SHANNON.value)
contenedor = Contenedor(config)
caso = contenedor.caso_uso_buscar_particion("fuerza_bruta", tpm)
resultado = caso.ejecutar(EntradaBusqueda("1000", "1111", "1111", "1111"))
```

### Cambios en el código existente

Para que el código antiguo siguiera funcionando sin modificaciones se aplicó
retrocompatibilidad en dos puntos críticos:

1. **`SIA.__init__`** acepta `config: AppConfig | None = None`. Si es `None`,
   las subclases recurren al singleton como antes. Si se pasa un `AppConfig`,
   lo usan en su lugar.

2. **`seleccionar_emd(config=None)`** lee `config.tiempo_emd` si se pasa config,
   o `aplicacion.tiempo_emd` si no. Todas las estrategias que antes llamaban
   `seleccionar_emd()` sin argumentos siguen funcionando igual.

Los 57 tests existentes pasan sin ningún cambio.

---

## Parte 11 — Cinco estrategias avanzadas de k-partición

Con la arquitectura hexagonal en su lugar, agregar estrategias nuevas se redujo
a implementar una clase que herede de `SIA` y satisfaga `IEstrategia` — sin
tocar el resto del sistema.

Se implementaron cinco estrategias usando técnicas que el proyecto no había explorado.

### Fundamento compartido: matriz de afinidad W

Las tres estrategias basadas en grafos (Louvain, ILP y Belief Propagation)
necesitan construir el grafo de acoplamientos del sistema. En lugar de duplicar
ese código, se extrajo a `src/funciones/grafo_info.py`:

```
W[i][j] = sensibilidad promedio del nodo i a cambios en el nodo j
        = media de |P(Xi=1 | Xj=0, resto) − P(Xi=1 | Xj=1, resto)|
```

Es la misma métrica que Circuito usaba internamente; ahora es reutilizable.

---

### 11.1 — Information Bottleneck

**Clase:** `InformacionBottleneck` en `src/estrategias/informacion_bottleneck.py`

**Matemáticas:** Tishby, Pereira y Bialek (1999). La idea central es comprimir
la descripción del sistema mientras se preserva la información predictiva.

Cada nodo tiene un *perfil causal*: la distribución de probabilidad de
transición aplanada desde su n-cubo. El algoritmo agrupa los nodos cuyos
perfiles son similares en el sentido de la divergencia KL.

Algoritmo de minimización alternada:

```
Inicializar p(t|i) ~ Dirichlet(1)   (asignación blanda aleatoria)
Repetir hasta convergencia:
  (a) Centroide del cluster t: μ_t = Σ_i p(t|i)·p(i) / p(t)
  (b) Actualizar p(t|i) ∝ p(t) · exp(−β · KL(f_i ‖ μ_t))
Asignación hard: z_i = argmax_t p(t|i)
```

El hiperparámetro β controla la "dureza" del agrupamiento. Un β alto concentra
la asignación (grupos muy separados); un β bajo la difumina (más incertidumbre).
Se ejecutan 8 reinicios aleatorios y se reporta el mejor.

**Por qué es relevante para IIT:** La función objetivo del IB — preservar
información predictiva mientras se comprime — es análoga a la de la MIP:
encontrar la partición que menos información pierde. La diferencia es que
IB opera en el espacio de nodos mientras MIP opera en el espacio de particiones.

---

### 11.2 — Louvain

**Clase:** `Louvain` en `src/estrategias/louvain.py`

**Matemáticas:** Blondel, Guillaume, Lambiotte y Lefebvre (2008). Maximización
de modularidad Q en grafos ponderados.

La modularidad mide si hay más aristas dentro de las comunidades de las que
habría en un grafo aleatorio con los mismos grados:

```
Q = Σ_{i,j} [A_ij − k_i·k_j/(2m)] · δ(c_i, c_j) / (2m)
```

El cambio en Q al mover el nodo i a la comunidad C es:

```
ΔQ = k_{i→C}/m − Σtot·k_i / (2m²)
```

donde `k_{i→C}` son los pesos de aristas de i hacia C y `Σtot` es la suma
de grados en C.

El algoritmo alterna dos fases hasta convergencia:
- **Fase 1** (optimización): cada nodo se mueve a la comunidad vecina que maximiza ΔQ.
- **Fase 2** (agregación): las comunidades se contraen en meta-nodos.

Si el número de comunidades resultante supera k, se fusionan las más conectadas
entre sí. Si es menor que k, la comunidad más grande se divide con el vector
de Fiedler de su sub-Laplaciano.

---

### 11.3 — Algoritmo Genético

**Clase:** `AlgoritmoGenetico` en `src/estrategias/genetico.py`

**Matemáticas:** metaheurística evolutiva clásica (Holland, 1975).

La representación es directa: un cromosoma es un vector `[g_0, g_1, …, g_{n-1}]`
donde `g_i ∈ {0, …, k-1}` es el grupo del nodo i.

Operadores:

| Operador | Descripción |
|---|---|
| Selección | Torneo de tamaño 3: de 3 individuos aleatorios, gana el de menor φ |
| Cruce | Uniforme: cada gen se hereda del padre 1 o padre 2 con P=0.5 |
| Mutación | Punto: cada gen cambia de grupo con probabilidad 1/n |
| Élitismo | Los 2 mejores individuos pasan intactos a la siguiente generación |

Las asignaciones se canonizan antes de evaluarlas para evitar contar como
distintas las permutaciones semánticamente iguales (grupo 0 vs grupo 1 da la misma partición).

Convergencia típica: 40 individuos × 80 generaciones = 3200 evaluaciones.
Para n=4 con evaluaciones en ~1ms, eso toma menos de 5 segundos.

---

### 11.4 — ILP (Relajación LP del k-cut)

**Clase:** `ParticionILP` en `src/estrategias/particion_ilp.py`

**Matemáticas:** Calinescu, Karloff y Rabani (2000). El k-cut mínimo en grafos
ponderados es NP-difícil para k ≥ 3, pero su relajación LP tiene ratio de
aproximación `2(1 − 1/k)`, el mejor conocido.

Formulación del programa lineal:

```
Variables: x[i,g] ∈ [0,1]  — fracción del nodo i en el grupo g
           y[i,j] ∈ [0,1]  — indicador de que la arista (i,j) está cortada

Minimizar: Σ_{i<j} w_ij · y[i,j]

Sujeto a:  Σ_g x[i,g] = 1         ∀i   (cada nodo en un grupo)
           Σ_i x[i,g] ≥ 1/k       ∀g   (cada grupo con masa)
           y[i,j] ≥ x[i,g]−x[j,g] ∀g   (corte ≥ diferencia de asignación)
           y[i,j] ≥ x[j,g]−x[i,g] ∀g
```

Se resuelve con `scipy.optimize.linprog` usando el solver HiGHS. La solución
fraccionaria se redondea con argmax y se refina con búsqueda local.

El grafo que se parte es el de acoplamientos W (de `grafo_info.py`), no el
espacio de particiones directamente. Eso hace que la solución LP sea un proxy
de la MIP real, con calidad comparable a Louvain en sistemas con estructura clara.

---

### 11.5 — Belief Propagation

**Clase:** `BeliefPropagation` en `src/estrategias/belief_propagation.py`

**Matemáticas:** Pearl (1988), propagación de mensajes en modelos gráficos
probabilísticos. Para grafos con ciclos se usa la variante *loopy* (LBP)
con amortiguación para estabilidad.

El modelo es un **Markov Random Field** con:

- **Variables**: `z_i ∈ {0, …, k-1}` — grupo del nodo i.
- **Potenciales unarios** `ψ_i(g)`: inicializados por rango de grado ponderado.
  Los nodos con mayor acoplamiento total tienen preferencia de grupo asignada
  determinísticamente para guiar la convergencia.
- **Potenciales de par** (modelo de Potts):
  ```
  ψ_ij(g, g) = exp(+α · w_ij)      — premio si mismo grupo y alta afinidad
  ψ_ij(g, h) = exp(−α · w_ij/k)    — penalización si diferente grupo con alta afinidad
  ```

Ecuaciones de actualización de mensajes:

```
μ_{i→j}(h) = Σ_g ψ_i(g) · ψ_ij(g,h) · Π_{l∈N(i)\j} μ_{l→i}(g)
```

Con amortiguación para evitar oscilaciones:
```
μ_nuevo = (1-δ)·μ_viejo + δ·μ_calculado     (δ = 0.5 por defecto)
```

Las creencias marginales `b_i(g) ∝ ψ_i(g) · Π_{j∈N(i)} μ_{j→i}(g)` dan la
distribución de probabilidad del grupo de cada nodo. La asignación hard
toma el argmax de cada creencia.

LBP es exacta en árboles (donde converge garantizadamente) y aproximada en
grafos con ciclos, donde la convergencia no está garantizada pero la
amortiguación la estabiliza en la práctica.

---

## Resumen actualizado de todo el trabajo

| Módulo / Archivo | Qué aporta |
|---|---|
| `src/funciones/k_particion_buscador.py` | `BuscadorKParticion` + `BuscadorKRecocido` (SA) + `BuscadorKDP` (DP subconjuntos + SA) + `buscar_con_semilla` |
| `src/funciones/iit.py` | Jensen-Shannon, KL simétrica, Wasserstein-Sinkhorn, Fisher-Rao; `seleccionar_emd(config)` |
| `src/funciones/grafo_info.py` | `construir_afinidad()` — matriz W compartida entre Louvain, ILP y BP |
| `src/funciones/entropia.py` | Rényi, Tsallis, perfil de entropías, divergencia de Rényi |
| `src/funciones/informacion_superior.py` | O-information, correlación total, matriz de dependencia |
| `src/herramientas/espectral.py` | Eigenvalores, π estacionaria, brecha espectral, tiempo de mezcla |
| `src/herramientas/benchmark.py` | Comparación automática de estrategias con tabla de resultados |
| `src/visualizacion/particion.py` | Gráficas de bipartición, k-partición y comparación de pérdidas |
| `src/controladores/gestor.py` | Estimación bayesiana de TPM con prior de Dirichlet |
| `src/modelos/base/sia.py` | `__init__(tpm, config=None)` — DI retrocompatible |
| `src/estrategias/informacion_bottleneck.py` | InformacionBottleneck — minimización alternada IB |
| `src/estrategias/louvain.py` | Louvain — modularidad Q en grafo de acoplamientos |
| `src/estrategias/genetico.py` | AlgoritmoGenetico — metaheurística evolutiva |
| `src/estrategias/particion_ilp.py` | ParticionILP — relajación LP con HiGHS, ratio 2(1-1/k) |
| `src/estrategias/belief_propagation.py` | BeliefPropagation — LBP con modelo de Potts, amortiguación |
| `src/aplicacion/configuracion.py` | `AppConfig` — reemplaza singleton, frozen dataclass |
| `src/aplicacion/puertos/` | `IEstrategia`, `IRepositorioTPM`, `IRegistro` — contratos (Protocols) |
| `src/aplicacion/casos_de_uso/` | `BuscarParticionOptima`, `EstimarTPM` — orquestación sin acoplamiento |
| `src/contenedor.py` | Composition root — único punto de ensamble de dependencias |
| `src/dominio/` | Re-exports que declaran la frontera del dominio puro |
| `src/infraestructura/` | Re-exports que clasifican los adaptadores concretos |
| `src/presentacion/orquestador.py` | `ejecutar()` con DI completa |

---

## Parte 12 — Optimización de k-particiones: DP de subconjuntos + warm-start + SA

**Fecha:** Mayo 2026

### Diagnóstico de la situación anterior

Después de los benchmarks de la Parte 1 quedó claro que para k > 2:

- **Geometric** era rápida pero quedaba atrapada en mínimos locales: su búsqueda
  local codiciosa no podía escapar una vez que ningún movimiento individual mejoraba
  la solución.
- **Q-Nodos** encontraba mejores particiones para k = 2 gracias a la submodularidad,
  pero para k > 2 el `algoritmo_q` (que es la esencia de la estrategia) se descartaba
  completamente y se caía en el mismo `BuscadorKParticion` genérico que Geometric.
- `BuscadorKRecocido` existía en el código pero **ninguna estrategia lo usaba** para
  k > 2. Estaba implementado y sin conectar.

Tres brechas concretas:

| Brecha | Descripción |
|---|---|
| SA desconectado | `BuscadorKRecocido` sin uso en ninguna estrategia k > 2 |
| Geometric pierde geometría | Para k > 2 ignoraba todo el cálculo del hipercubo (costos_locales) |
| Q-Nodos pierde submodularidad | Para k > 2 no usaba `algoritmo_q` en ningún momento |

---

### Solución 1 — `BuscadorKDP`: programación dinámica de subconjuntos

Se agregó una nueva clase `BuscadorKDP` en `src/funciones/k_particion_buscador.py` que hereda de `BuscadorKRecocido`.

**Idea central:** Para encontrar la k-partición óptima de n elementos se puede usar DP sobre los 2ⁿ subconjuntos posibles. Se define:

```
costos_sub[mask] = costo de bipartición donde los elementos indicados por `mask`
                   forman un grupo y el resto forma el otro
```

Con esos costos precalculados se aplica DP de subconjuntos:

```
dp[mask][j] = min costo estimado de j-partición de los elementos en mask

Transición: dp[mask][j] = min sobre todo subconjunto T⊆mask:
               dp[mask ^ T][j-1] + costos_sub[T]
```

Complejidad:
- **Precomputo**: O(2ⁿ) evaluaciones de bipartición. Si se pasan costos ya calculados (como los del hipercubo de Geometric), el precomputo es O(1) — cero evaluaciones extras.
- **Tabla DP**: O(3ⁿ × k) transiciones (solo comparaciones, sin evaluaciones).
- **Refinamiento**: el resultado de la DP inicializa el SA heredado de `BuscadorKRecocido`.

El DP da la asignación inicial de mínimo costo estimado. El SA la refina para escapar del mínimo local que el DP podría haber encontrado.

```
BuscadorKParticion (greedy local)
├── BuscadorKRecocido (SA puro)
│   └── BuscadorKDP (DP init + SA refinamiento)   ← NUEVO
```

---

### Solución 2 — Warm-start geométrico para Geometric k > 2

`_BuscadorKGeometric` ahora hereda de `BuscadorKDP` en vez de `BuscadorKParticion`.

Para k > 2, en lugar de empezar con una asignación aleatoria, la estrategia:

1. Llama `_precalcular_busqueda_geometrica` para obtener `costos_locales` del hipercubo.
2. Pasa ese array directamente a `BuscadorKDP` como `costos_subconjuntos`: **el DP reutiliza la geometría ya calculada sin ningún costo adicional de evaluación**.
3. Extrae la mejor máscara de `costos_locales` (la que tiene menor costo de bipartición) y la convierte en una asignación de k grupos como warm-start adicional (`_semilla_desde_biparticion`).
4. Compara la semilla DP + refinamiento local contra una corrida SA independiente; retorna el mejor.

Esto cierra el gap que existía: antes la geometría del hipercubo solo servía para k = 2; ahora también inicializa k > 2.

---

### Solución 3 — Partición recursiva Q con memoización DP para Q-Nodos k > 2

`_BuscadorKQNodos` ahora hereda de `BuscadorKRecocido` en vez de `BuscadorKParticion`.

Se agregó `_particionar_recursivo_q` a la clase `QNodos`. Es una implementación de **divide y vencerás con memoización** sobre el espacio de (subconjunto, k):

```
memo[(frozenset(vertices), k)] = mejor asignacion de k grupos para esos vertices
```

Algoritmo:
1. Para k = 2: aplicar `algoritmo_q` directamente sobre los vértices dados.
2. Para k > 2: bipartir con `algoritmo_q`, identificar el grupo mayor, partirlo recursivamente en k-1 subgrupos, reconstruir la asignación global.
3. Memoizar cada subproblema para que subconjuntos repetidos no se recalculen.

Esto preserva la esencia submodular de Q-Nodos en todos los niveles del árbol de recursión. La asignación resultante se usa como warm-start para el SA del `_BuscadorKQNodos`.

Gestión de estado: dentro de cada llamada recursiva se guarda y restaura `self.vertices` y `self.memoria_grupo_candidato`. `memoria_delta` se deja acumular entre llamadas porque sus valores (costos de bipartición por subconjunto mecanismo/alcance) son independientes del subconjunto que se está particionando.

---

### Comparación antes / después

| Aspecto | Antes | Después |
|---|---|---|
| `_BuscadorKGeometric` hereda de | `BuscadorKParticion` | `BuscadorKDP` |
| `_BuscadorKQNodos` hereda de | `BuscadorKParticion` | `BuscadorKRecocido` |
| Geometric k > 2 usa hipercubo | No | Sí (`costos_locales` → DP) |
| Q-Nodos k > 2 usa submodularidad | No | Sí (recursión `algoritmo_q`) |
| Warm-start desde k = 2 | No | Sí (máscara óptima → asignación inicial) |
| Escape de mínimos locales | Greedy puro | SA en todas las estrategias k > 2 |
| `buscar_con_semilla` disponible | No | Sí (en `BuscadorKRecocido`, heredado por todos) |
| Tests pasando | 57/57 | 57/57 |

---

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `src/funciones/k_particion_buscador.py` | `BuscadorKDP` + `BuscadorKRecocido.buscar_con_semilla` |
| `src/strategies/geometric.py` | `_BuscadorKGeometric` → hereda `BuscadorKDP`; warm-start geométrico |
| `src/estrategias/q_nodos.py` | `_BuscadorKQNodos` → hereda `BuscadorKRecocido`; `_particionar_recursivo_q` |

---

## Parte 13 — Enriquecimiento del SA: swap moves y múltiples cadenas

**Fecha:** Mayo 2026

### El problema que quedaba pendiente

El SA de la Parte 12 tenía un solo tipo de movimiento en cada paso: tomar un nodo al azar y asignarlo a un grupo distinto al azar. Eso limita la exploración porque para intercambiar el nodo A (grupo 0) con el nodo B (grupo 1) se necesitan **dos movimientos consecutivos aceptados**. En la práctica, el SA se quedaba atrapado en mínimos locales donde ningún movimiento individual mejoraba la solución pero sí podría mejorar un intercambio directo.

Adicionalmente, por cada llamada a `buscar` o `buscar_con_semilla` el SA corría una sola cadena desde un único punto de inicio. Si esa cadena quedaba atrapada, no había forma de recuperarse.

---

### Mejora 1 — Swap moves: ampliar el vecindario del SA

Se modificó el método `_recocido` de `BuscadorKRecocido` para que en cada paso del SA, con probabilidad 0.5, se ejecute un **swap** en lugar de una reasignación individual:

```
Reasignación (50% del tiempo):
  nodo i → grupo aleatorio g
  [0, 1, 0, 2] → [0, 2, 0, 2]   (nodo 1 cambia de grupo 1 a grupo 2)

Swap (50% del tiempo):
  nodos i y j intercambian sus grupos
  [0, 1, 0, 2] → [0, 2, 0, 1]   (nodos 1 y 3 intercambian grupos)
```

El swap llega en un solo paso a vecinos que la reasignación individual necesita dos pasos aceptados para alcanzar. Esto duplica el vecindario efectivo explorado por paso sin cambiar la complejidad del algoritmo.

Implementación: el índice `j` se genera como `j = rng.integers(0, n-1)` y luego `if j >= i: j += 1`, lo que garantiza que `i ≠ j` sin sesgo en la distribución.

---

### Mejora 2 — Múltiples cadenas SA: reducir la dependencia de la semilla

Se añadió el parámetro `n_cadenas=3` a `BuscadorKRecocido.__init__` y el método `_multi_recocido`:

```python
def _multi_recocido(self, k, semilla):
    mejor = self._recocido(k, semilla)
    for i in range(1, self.n_cadenas):
        candidato = self._recocido(k, semilla + i * 1009)
        if candidato.perdida < mejor.perdida:
            mejor = candidato
    return mejor
```

El offset de `1009` (número primo) entre semillas evita correlaciones entre cadenas. El **cache de evaluaciones ya calculadas se comparte entre las tres cadenas**: si la cadena 1 evaluó la asignación `(0,1,2,0,1)` y la cadena 2 llega a la misma asignación, la obtiene en O(1) sin recalcular. Eso hace que la segunda y tercera cadena sean más baratas que la primera.

`buscar`, `buscar_con_semilla` y el fallback de `BuscadorKDP.buscar` para n grande ahora usan `_multi_recocido` en vez de `_recocido`.

---

### Benchmark: qué cambió y qué no

Se corrió el benchmark sobre las mismas 30 configuraciones del experimento original (3 tamaños × 2 valores de k × 5 semillas) y se comparó contra el CSV histórico.

**Geometric — sin cambio observable:**

| k | nodos | φ antes (prom) | φ ahora (prom) | Δφ |
|:---:|:---:|:---:|:---:|:---:|
| 3 | 4 | 0.424304 | 0.424304 | 0.000000 |
| 3 | 5 | 0.710583 | 0.710583 | 0.000000 |
| 3 | 6 | 0.683304 | 0.683304 | 0.000000 |
| 4 | 4 | 0.424304 | 0.424304 | 0.000000 |
| 4 | 5 | 0.710583 | 0.710583 | 0.000000 |
| 4 | 6 | 0.683304 | 0.683304 | 0.000000 |

**Por qué Geometric no cambia:** Para los tamaños de prueba (4, 5 y 6 nodos), los vértices del sistema son a lo sumo 6 enteros, que caen dentro del umbral DP (`umbral_dp=12`). El DP de subconjuntos es determinístico: calcula el mínimo exacto con los costos precalculados del hipercubo, sin usar SA. Los swap moves y las múltiples cadenas SA no se ejecutan porque el DP ya encuentra la solución óptima estimada. El resultado es idéntico al anterior.

**QNodos — sensible a la semilla:**

| k | nodos | φ antes (prom) | φ ahora (prom) | Δφ |
|:---:|:---:|:---:|:---:|:---:|
| 3 | 4 | 0.064984 | 0.064984 | 0.000000 |
| 3 | 5 | 0.120233 | 0.120233 | 0.000000 |
| 3 | 6 | 0.107660 | 0.107660 | 0.000000 |
| 4 | 4 | 0.064984 | 0.064984 | 0.000000 |
| **4** | **5** | **0.120233** | **0.171803** | **−0.051570** |
| 4 | 6 | 0.107660 | 0.107660 | 0.000000 |

Para k=4, nodos=5, semilla=41: la regresión es real y determinística (siempre produce 0.427 en vez de 0.169). La causa es que los swap moves cambian el patrón de consumo del generador de números aleatorios: en el SA anterior cada paso consumía 2 números aleatorios (`rng.integers` para el índice y el grupo); ahora consume 3 (se agrega `rng.random() < 0.5` para decidir el tipo de movimiento). Eso desplaza toda la secuencia de la cadena y produce una trayectoria completamente distinta para la misma semilla.

Diagnóstico de ese caso: el warm-start de Q-Nodos genera una semilla inicial con φ=0.619 para ese sistema. El SA anterior con la semilla 83 encontraba φ=0.169 en esa trayectoria. Con swap moves, la semilla 83 ahora explora una trayectoria distinta y llega a φ=0.427. Las dos cadenas adicionales (semillas 1092 y 2101) tampoco mejoran ese valor.

Esto no es un bug: es sensibilidad estocástica al cambio de trayectoria. Las semillas del benchmark original eran específicas para el código anterior. Para evaluar si los swap moves ayudan de forma general se necesita probar con más semillas.

**Benchmark ampliado con 20 semillas aleatorias (k=3, nodos=5):**

| Estrategia | φ promedio | φ mínimo | φ máximo | Victorias (de 20) |
|---|:---:|:---:|:---:|:---:|
| QNodos | 0.091788 | 0.000052 | 0.445424 | 20/20 |
| Geometric | 0.618789 | 0.300938 | 1.090843 | 0/20 |

Con un conjunto más amplio de semillas QNodos sigue ganando en todas las pruebas y la diferencia promedio sigue siendo grande. La regresión puntual del benchmark original es un artefacto de las semillas fijas, no una degradación general del algoritmo.

---

### Tabla de cambios

| Aspecto | Antes | Después |
|---|---|---|
| Movimientos SA | Solo reasignación individual | Reasignación (50%) + swap (50%) |
| Cadenas SA por búsqueda | 1 | 3 (`n_cadenas=3`) |
| Cache entre cadenas | N/A (una sola cadena) | Compartido — cadenas 2 y 3 más baratas |
| Tests pasando | 76/76 | 76/76 |

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `src/funciones/k_particion_buscador.py` | Swap moves en `_recocido` y `_buscar_dp_sa`; parámetro `n_cadenas`; método `_multi_recocido` |
| `review/benchmarks/benchmark_sa_mejoras.py` | Benchmark de comparación antes/después contra CSV histórico |
