# Bitácora de trabajo: K-Particiones y Estrategia Circuito

**Proyecto:** Partición de Mínima Pérdida de Información (MIP) — IIT  
**Fecha:** Abril 2026  
**Autor:** CamiOso

---

## ¿Por qué se hizo esto?

El proyecto ya tenía varias estrategias para dividir un sistema en dos grupos
y encontrar la división que menos información pierde. Pero eso es una limitación
fuerte: no todos los sistemas se dividen bien en exactamente dos partes.

Piensen en un equipo de fútbol. Si les digo "divídanlo en dos grupos", me pueden
dar defensas vs. atacantes. Pero si el equipo en realidad funciona en tres bloques
(defensa, mediocampo, delantera), forzar dos grupos no va a reflejar bien cómo
trabajan juntos.

Lo mismo pasa con los sistemas de nodos que estudia este proyecto. Entonces la
tarea fue:

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

## Resumen de todo el trabajo

| Módulo / Archivo | Qué aporta |
|---|---|
| `src/funciones/k_particion_buscador.py` | `BuscadorKParticion` (Template Method) + `BuscadorKRecocido` (SA) |
| `src/funciones/iit.py` | Jensen-Shannon, KL simétrica, Wasserstein-Sinkhorn, Fisher-Rao |
| `src/funciones/entropia.py` | Rényi, Tsallis, perfil de entropías, divergencia de Rényi |
| `src/funciones/informacion_superior.py` | O-information, correlación total, matriz de dependencia |
| `src/herramientas/espectral.py` | Eigenvalores, π estacionaria, brecha espectral, tiempo de mezcla |
| `src/herramientas/benchmark.py` | Comparación automática de estrategias con tabla de resultados |
| `src/visualizacion/particion.py` | Gráficas de bipartición, k-partición y comparación de pérdidas |
| `src/controladores/gestor.py` | Estimación bayesiana de TPM con prior de Dirichlet |
