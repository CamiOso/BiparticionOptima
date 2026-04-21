# Informe del Proyecto: Búsqueda de la Partición Óptima de un Sistema

## ¿De qué trata este proyecto?

Imagina un grupo de personas que trabajan juntas. Si divides el grupo en dos equipos,
inevitablemente se pierde algo de coordinación entre ellos. La pregunta es:
**¿cómo dividirlos de tal forma que se pierda lo menos posible?**

Eso es exactamente lo que hace este proyecto, pero con sistemas de nodos
(pueden ser neuronas, sensores, interruptores, etc.).

El problema se llama **MIP** (Minimum Information Partition, o Partición de Mínima
Pérdida de Información), y viene de la **Teoría de Información Integrada (IIT)**,
que estudia qué tan "unido" o "integrado" está un sistema.

---

## Paso 1 — El sistema: nodos que se influyen entre sí

Un sistema tiene **n nodos**. Cada nodo puede estar en estado 0 (apagado) o 1 (encendido).
En cada instante de tiempo, el estado de cada nodo cambia según los estados de los demás.

**Ejemplo con 3 nodos (A, B, C):**

```
Estado actual:  A=0, B=0, C=0  →  "000"
Estado siguiente: ¿cuál es la probabilidad de que A=1?, ¿B=1?, ¿C=1?
```

Esas probabilidades están guardadas en la **TPM** (Transition Probability Matrix —
Matriz de Probabilidades de Transición).

### ¿Cómo se lee la TPM?

La TPM tiene una fila por cada posible estado del sistema (`2^n` filas) y una
columna por cada nodo. Cada celda dice: "si el sistema está en este estado,
¿con qué probabilidad este nodo será 1 en el siguiente instante?"

**Ejemplo de TPM para 3 nodos** (formato: estado → [P(A=1), P(B=1), P(C=1)]):

```
000 → [0.0, 0.0, 0.0]   (ninguno se enciende)
001 → [0.0, 0.0, 1.0]   (solo C sigue encendido)
010 → [0.0, 1.0, 0.0]   (solo B sigue encendido)
011 → [0.0, 1.0, 1.0]
100 → [1.0, 0.0, 0.0]
101 → [1.0, 0.0, 1.0]
110 → [1.0, 1.0, 0.0]
111 → [1.0, 1.0, 1.0]   (todos siguen encendidos)
```

---

## Paso 2 — El hipercubo: visualizar el sistema

Con 3 nodos y 2 estados posibles cada uno, hay `2^3 = 8` estados en total.
Esos 8 estados se pueden organizar como los vértices de un **cubo** (hipercubo de
dimensión 3), donde dos estados son vecinos si difieren en exactamente un bit.

```
    011 ---- 111
   /|        /|
  010 ---- 110 |
  |  |     |  |
  | 001 ---| 101
  |/        |/
  000 ---- 100
```

Cada **arista** del cubo conecta dos estados que difieren en un solo nodo.
Eso representa un "paso elemental" de cambio en el sistema.

### Tabla de costos entre estados (generada por el proyecto)

El proyecto calcula el costo de ir de un estado a otro usando la fórmula
`γ = 2^(-d)` donde `d` es la distancia Hamming (cuántos bits difieren).

**Tabla de costos reales (`tabla_costos_3_variables.csv`):**

| estado | 000  | 001  | 010  | 011  | 100  | 101  | 110  | 111  |
|--------|------|------|------|------|------|------|------|------|
| 000    | 1.00 | 0.50 | 0.50 | 0.25 | 0.50 | 0.25 | 0.25 | 0.13 |
| 001    | 0.50 | 1.00 | 0.25 | 0.50 | 0.25 | 0.50 | 0.13 | 0.25 |
| 010    | 0.50 | 0.25 | 1.00 | 0.50 | 0.25 | 0.13 | 0.50 | 0.25 |
| 011    | 0.25 | 0.50 | 0.50 | 1.00 | 0.13 | 0.25 | 0.25 | 0.50 |
| 100    | 0.50 | 0.25 | 0.25 | 0.13 | 1.00 | 0.50 | 0.50 | 0.25 |
| 101    | 0.25 | 0.50 | 0.13 | 0.25 | 0.50 | 1.00 | 0.25 | 0.50 |
| 110    | 0.25 | 0.13 | 0.50 | 0.25 | 0.50 | 0.25 | 1.00 | 0.50 |
| 111    | 0.13 | 0.25 | 0.25 | 0.50 | 0.25 | 0.50 | 0.50 | 1.00 |

**Lectura:** costo de `000 → 001` = 0.50 (difieren en 1 bit → `2^(-1) = 0.5`).
Costo de `000 → 111` = 0.125 (difieren en 3 bits → `2^(-3) = 0.125`).

---

## Paso 3 — Las proyecciones del hipercubo

Cuando se hace una **partición** del sistema (dividir los nodos en grupos),
se proyecta el hipercubo sobre dimensiones más pequeñas. Cada proyección
agrupa estados que comparten el mismo valor en las dimensiones proyectadas.

**Proyecciones reales del cubo de 3 variables (`proyecciones_3_variables.csv`):**

| Proyección | Grupo | Estados agrupados |
|------------|-------|-------------------|
| AB         | 00    | 000, 001           |
| AB         | 01    | 010, 011           |
| AB         | 10    | 100, 101           |
| AB         | 11    | 110, 111           |
| AC         | 00    | 000, 010           |
| AC         | 01    | 001, 011           |
| BC         | 00    | 000, 100           |
| BC         | 01    | 001, 101           |

**Lectura:** si ignoramos el nodo C (proyección AB), los estados 000 y 001
son indistinguibles (ambos tienen A=0, B=0). La partición "funde" esos dos estados.

---

## Paso 4 — La pérdida de información (EMD)

Cuando partimos el sistema en dos grupos, cada grupo "pierde visibilidad" sobre el otro.
Esa pérdida se mide con la **EMD** (Earth Mover's Distance — Distancia del Transportador
de Tierra): cuánto esfuerzo cuesta "mover" una distribución de probabilidad para que
se parezca a otra.

**Idea simple:** imagina dos montones de arena de formas distintas. La EMD mide cuánta
arena hay que mover (y qué tan lejos) para que un montón tenga la misma forma que el otro.

- **Distribución del subsistema:** lo que el sistema predice cuando está completo.
- **Distribución de la partición:** lo que predice cuando los grupos están desconectados.
- **Pérdida (φ):** la EMD entre esas dos distribuciones. Si φ = 0, la partición no
  rompe nada. Cuanto mayor sea φ, más "integrado" está el sistema en ese punto.

**Objetivo:** encontrar la partición donde φ sea **mínima** (el "punto débil" del sistema).

---

## Paso 5 — Las estrategias implementadas

Se implementaron cinco formas de encontrar esa partición mínima:

### Estrategia 1: Fuerza Bruta (referencia exacta)

**Idea:** prueba absolutamente todas las particiones posibles y devuelve la que
tiene menor φ.

**Problema:** el número de particiones crece exponencialmente. Para n=8 nodos
hay miles de particiones posibles. Para n=20, es imposible en tiempo razonable.

**Complejidad:** `O(2^n)` — muy lento para sistemas grandes.

---

### Estrategia 2: Phi (con PyPhi)

**Idea:** usa la librería académica PyPhi si está instalada. Si no, usa una
heurística interna que prueba un conjunto reducido de particiones candidatas.

---

### Estrategia 3: Geometric (sobre el hipercubo)

**Idea:** en lugar de probar todas las particiones a ciegas, usa la estructura
geométrica del hipercubo para guiar la búsqueda.

**Analogía:** en vez de buscar la salida de un laberinto tocando todas las paredes,
usas un mapa (la geometría del cubo) para ir directamente a las zonas más prometedoras.

Tiene dos modos:
- **Estricto:** solo usa la tabla recursiva del hipercubo → `O(n · 2^n)`.
  Es más rápido pero puede perder precisión.
- **Refinado:** agrega refinamiento local (hill-climbing) y reinicios aleatorios
  para mejorar la solución encontrada por el modo estricto.

---

### Estrategia 4: Q-Nodos (submodular)

**Idea:** usa una propiedad matemática llamada **submodularidad** — si agregas
nodos a un grupo, la pérdida crece de forma decreciente (cada nodo adicional
aporta menos que el anterior). Esto permite una búsqueda eficiente tipo "greedy".

**Analogía:** como armar un equipo de trabajo. El primer experto aporta mucho;
el segundo algo menos; el décimo ya casi no cambia nada.

---

### Estrategia 5: Circuito (nueva — red eléctrica espectral)

**Idea:** modela el sistema como un **circuito eléctrico** donde cada conexión
entre nodos tiene una **conductancia** proporcional al acoplamiento entre ellos
(medido desde la TPM). Nodos muy acoplados tienen alta conductancia (como un cable
grueso); nodos poco relacionados tienen baja conductancia (cable delgado o cortado).

El **Laplaciano** del grafo captura esta estructura. Sus eigenvectores revelan
los "cortes naturales" del sistema:

- **k=2 (bipartición):** el **vector de Fiedler** (segundo eigenvector del Laplaciano)
  divide los nodos en dos grupos según el signo de sus componentes. Es la partición
  que minimiza el flujo entre los grupos — exactamente el "punto débil".

- **k>2 (k-partición):** usa los primeros k eigenvectores como coordenadas en un
  espacio reducido, y aplica k-means para agrupar los nodos.

**Complejidad:** `O(n³)` — determinista, sin búsqueda aleatoria, solo álgebra lineal.

**Analogía con electricidad:**
```
Sistema integrado:      Sistema particionable:
  A ═══ B ═══ C           A ═══ B    C
  (cables gruesos)         (cable fino entre B y C)
  → Difícil de partir      → Fácil de partir aquí
```

---

## Paso 6 — Resultados experimentales

### 6.1 Geometric vs Fuerza Bruta

Se compararon las tres estrategias en sistemas de 5 a 8 nodos, con 3 semillas
aleatorias por tamaño. Datos reales del benchmark:

| Nodos | Speedup Estricto | Speedup Refinado | Error φ Estricto | Error φ Refinado |
|-------|-----------------|------------------|-----------------|-----------------|
| 5     | 7.9x            | 1.3x             | 0.449           | 0.000           |
| 6     | 17.0x           | 10.8x            | 0.396           | 0.004           |
| 7     | 41.9x           | 29.7x            | 0.598           | 0.000           |
| 8     | 107.1x          | 39.7x            | 0.617           | 0.000           |

**Lectura:**
- El modo **estricto** es hasta 107 veces más rápido que Fuerza Bruta para 8 nodos,
  pero sacrifica precisión (error φ ≈ 0.62).
- El modo **refinado** sigue siendo 40 veces más rápido y prácticamente no pierde
  precisión (error φ ≈ 0.00 en la mayoría de casos).

### 6.2 Tiempos absolutos de referencia (8 nodos)

| Estrategia          | Tiempo promedio |
|---------------------|----------------|
| Fuerza Bruta        | 81.3 segundos  |
| Geometric Estricto  | 0.77 segundos  |
| Geometric Refinado  | 2.07 segundos  |

### 6.3 Q-Nodos vs Geometric en k-particiones

Se probaron k=3 y k=4 grupos en sistemas de 4, 5 y 6 nodos (5 semillas cada uno):

| k | Nodos | Speedup Geometric sobre QNodos | Victorias en φ (QNodos) | Victorias en φ (Geometric) |
|---|-------|-------------------------------|------------------------|---------------------------|
| 3 | 4     | 48.2x                         | 5/5                    | 0/5                       |
| 3 | 5     | 22.1x                         | 5/5                    | 0/5                       |
| 3 | 6     | 10.1x                         | 5/5                    | 0/5                       |
| 4 | 4     | 112.4x                        | 5/5                    | 0/5                       |

**Lectura:** Geometric es mucho más rápido, pero Q-Nodos encuentra particiones
con menor φ en todos los casos probados para k≥3. Es el clásico tradeoff
**velocidad vs. precisión**.

### 6.4 Optimización para sistemas grandes (n ≥ 9)

Para sistemas de 9 y 10 nodos se activó la optimización (muestreo de máscaras,
simetrías del hipercubo, paralelización):

| Nodos | Speedup con optimización | Error φ promedio |
|-------|--------------------------|-----------------|
| 9     | 1.05x                    | 0.006           |
| 10    | 1.22x                    | 0.000           |

La optimización mantiene la precisión mientras reduce el tiempo de cómputo.

---

## Paso 7 — ¿Cómo se conectan todas las piezas?

```
TPM (datos del sistema)
        │
        ▼
  Sistema → n-cubos (un cubo por nodo)
        │
        ▼
  Condicionamiento (fija nodos de fondo)
  Marginalización (ignora nodos irrelevantes)
        │
        ▼
  Distribución del subsistema completo
        │
        ├──► Fuerza Bruta: prueba todas las particiones
        ├──► Geometric:    busca en la geometría del hipercubo
        ├──► Q-Nodos:      búsqueda submodular greedy
        └──► Circuito:     eigendescomposición del Laplaciano
                │
                ▼
        Partición con menor φ (MIP)
```

---

## Conclusiones

1. **El problema es difícil:** en su versión exacta (Fuerza Bruta) crece
   exponencialmente con el número de nodos.

2. **Geometric modo refinado** es la mejor relación velocidad-precisión:
   hasta 40x más rápido que Fuerza Bruta con error prácticamente nulo.

3. **Q-Nodos** encuentra mejores particiones en k≥3 grupos, pero es más lento.

4. **Circuito** aporta un enfoque completamente distinto: en vez de buscar
   particiones por ensayo y error, infiere la estructura del sistema desde
   el álgebra del grafo. Es determinista y escala en O(n³).

5. Todas las estrategias comparten la misma interfaz base (`SIA`) y retornan
   el mismo objeto `Solucion`, lo que permite compararlas directamente.

---

## Cómo reproducir los resultados

```bash
# Instalar entorno
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ejecutar todas las estrategias
python exec.py

# Benchmark Geometric vs Fuerza Bruta
PYTHONPATH=. python review/benchmarks/benchmark_geometric.py

# Ejemplo de 3 variables con tabla de costos
PYTHONPATH=. python review/benchmarks/ejemplo_3_variables.py

# Visualizaciones del hipercubo
PYTHONPATH=. python review/benchmarks/visualizacion_3_variables.py

# Usar la nueva estrategia Circuito
python -c "
from src.estrategias.circuito import Circuito
import numpy as np
tpm = np.random.rand(8, 3).astype(np.float32)
c = Circuito(tpm)
sol = c.aplicar_estrategia('101', '111', '111', '111', k=2)
print(sol)
"
```
