# Pseudocódigos macro: k-particiones y estrategia de circuitos

**Proyecto:** Partición de Mínima Pérdida de Información (MIP) — IIT  
**Fecha:** Mayo 2026  
**Origen:** Diseño teórico en lienzo en blanco, sin pensar en estructuras de datos primero

---

## El cambio filosófico de bipartición a k-partición

En bipartición la pregunta es: *¿cuál es el único corte que más divide al sistema?*  
En k-partición la pregunta es: *¿cuál es la **jerarquía completa** de divisiones del sistema?*

Esta diferencia cambia todo. La solución no es "buscar k grupos" — es **descubrir la estructura de división natural del sistema** y leer k-particiones de ella.

La clave de pensar en lienzo en blanco: no comenzar con estructuras de datos (qué arreglo, qué hash map, qué indexación), sino con la pregunta de qué estructura tiene el **problema**. Cuando uno piensa en estructuras de datos primero, mutila la estrategia porque restringe el espacio de ideas antes de entenderlo.

---

## Algoritmo 1 — Geométrica K: árbol de cortes divisivo

### Insight macro

Cada corte mínimo dentro de un componente produce dos hijos. Si se guardan todos los cortes como un árbol binario (dendrograma), la k-partición óptima ya está ahí — es solo leer las k hojas del árbol.

No se "busca k grupos". Se **construye la jerarquía de divisiones** y se lee cualquier k de ella.

### Por qué funciona

Si la función de pérdida EMD es submodular (y lo es), entonces el orden greedy de cortes mínimos produce la jerarquía óptima de particiones. Esto es análogo al árbol de Gomory-Hu para flujos de red: una sola estructura resuelve todos los problemas de corte simultáneamente.

### Pseudocódigo

```
ALGORITHM GeometricK(sistema, k_max):

// PREPARACIÓN (igual que la versión de bipartición)
grafo = construir_grafo_causal(sistema)
eliminar_aristas_perdida_cero(grafo)   // aristas irreducibles: no aportan al corte

// FASE 1: CONSTRUIR ÁRBOL DE DIVISIONES (dendrograma)
árbol = Árbol(raíz = todos_los_nodos)
cola  = cola_prioridad([raíz])   // priorizar componentes más grandes primero

MIENTRAS árbol.hojas() < k_max:
    componente = cola.extraer()
    SI |componente| == 1: continuar   // componente atómico, no se puede dividir

    // Encontrar el corte que menos información pierde dentro de este componente
    (izq, der, pérdida) = corte_mínimo_EMD(subgrafo(componente))

    // Registrar la división en el árbol
    árbol.dividir(componente → izq, der, pérdida)

    // Ambos hijos son candidatos a seguir dividiéndose
    cola.insertar(izq)
    cola.insertar(der)

// FASE 2: LEER TODAS LAS K-PARTICIONES DEL ÁRBOL
// El árbol ya tiene toda la información — solo se lee en distintos niveles
PARA k en 2..árbol.hojas():
    partición_k = árbol.hojas_en_nivel(k)
    Φ_k         = EMD(sistema, partición_k)
    registrar(k, partición_k, Φ_k)

// FASE 3: SELECCIONAR K ÓPTIMA
// La MIP es la que minimiza Φ sobre todas las k evaluadas
RETORNAR argmin_k(Φ_k), su partición y pérdida
```

### Lo nuevo vs. bipartición

Antes el algoritmo terminaba en la primera desconexión (la primera vez que el grafo se partía en dos). Ahora *continúa dentro de cada componente* de forma recursiva, construyendo el árbol completo. La k-partición no se busca — emerge como nivel k del árbol.

### Complejidad

| Fase | Operación | Costo |
|---|---|---|
| Corte mínimo por componente | EMD + DFS | O(n² · log n) por nivel |
| Árbol completo | k_max niveles | O(k_max · n² · log n) |
| Lectura de la k-partición | Recorrido del árbol | O(n) por k |

---

## Algoritmo 2 — QNodes K: Queyranne revela todas las k-particiones

### Insight macro

El algoritmo de Queyranne no produce solo UNA bipartición — produce una **secuencia de n-1 contracciones ordenadas por fuerza de corte**. Esta secuencia forma un **árbol de contracciones** (análogo al árbol de Gomory-Hu). La k-partición óptima se obtiene eligiendo cuáles k-1 contracciones NO hacer: esas son las "fronteras naturales" del sistema.

### Por qué funciona

Queyranne basa su correctitud en la submodularidad de EMD. La secuencia de contracciones que produce codifica todos los cortes posibles en orden de costo. Las k-1 contracciones más baratas son exactamente los k-1 cortes que menos información destruyen. No contraerlos es equivalente a hacer esos cortes.

### Pseudocódigo

```
ALGORITHM QNodesK(sistema, k):

// FASE 1: Construir N-cubo (igual que antes)
ncubo = construir_Ncubo(sistema.TPM)

// FASE 2: Queyranne COMPLETO — obtener árbol de contracciones
// A diferencia de la versión de bipartición, NO se para al llegar a 2 activos.
// Se deja correr hasta el final para obtener la estructura completa.
secuencia_contracciones = []
activos = todos_los_elementos

MIENTRAS |activos| > 1:
    // Maximum Adjacency Ordering da el "par pendiente" (pendant pair)
    (s, t, valor_corte) = MaxAdjOrdering(activos, ncubo)

    secuencia_contracciones.agregar((s, t, valor_corte))
    fusionar(activos, s, t)   // contrae s y t en un meta-nodo

// secuencia tiene exactamente n-1 entradas → árbol de contracciones completo

// FASE 3: ENCONTRAR K-PARTICIÓN ÓPTIMA
// Las k-1 contracciones con MENOR valor de corte son los "eslabones débiles"
// del sistema: los lugares donde más naturalmente se divide.
// NO contraer esos k-1 = separar el sistema en k grupos naturales.

ordenadas = ordenar(secuencia_contracciones, por=valor_corte, ascendente)
prohibidas = conjunto(ordenadas[0..k-2])   // las k-1 más débiles NO se contraen

// Reproducir la secuencia de contracciones, omitiendo las prohibidas
grupos = {{v} para cada v en sistema}   // cada nodo empieza en su propio grupo

PARA (s, t, val) en secuencia_contracciones:
    SI (s, t, val) NO está en prohibidas:
        fusionar(grupos, grupo_de(s), grupo_de(t))

// Al terminar, grupos tiene exactamente k componentes
pérdida_total = suma(valor_corte de cada contracción en prohibidas)

RETORNAR grupos, pérdida_total
```

### Lo nuevo vs. bipartición

Antes Queyranne terminaba al llegar a 2 activos: devolvía un único corte. Ahora se deja correr hasta el final, se guarda el árbol entero, y se elige cuáles contracciones "no hacer" para obtener k grupos. El algoritmo de bipartición es un caso especial con k=2.

### Complejidad

| Fase | Operación | Costo |
|---|---|---|
| Queyranne completo | n-1 iteraciones de MaxAdjOrdering | O(n³) |
| Seleccionar k-1 eslabones débiles | Ordenar n-1 entradas | O(n log n) |
| Reproducir contracciones | Recorrido de la secuencia | O(n) |
| **Total** | | **O(n³)** — igual que la bipartición |

El punto crítico: resolver k-particiones cuesta lo mismo que resolver bipartición.

---

## Algoritmo 3 — Estrategia de Circuitos (nueva)

### Insight fundamental (lienzo en blanco)

La información integrada nace de los **ciclos causales**: A causa B que causa C que causa A. Un sistema es indivisible cuando sus ciclos conectan todo. Partir el sistema es *romper ciclos*.

Por lo tanto: **la unidad atómica de información integrada no es el nodo, ni la arista, ni el subconjunto — es el circuito (ciclo dirigido)**.

Si se puede asignar cada ciclo del sistema a un grupo de la partición (sin que ningún ciclo quede partido entre dos grupos), la pérdida es cero. La pérdida de una partición es proporcional a cuántos ciclos rompe y cuán "fuertes" son esos ciclos.

Esto convierte el problema en: **particionar un hipergrafo donde los hiperarcos son los circuitos**.

### Por qué es distinto en esencia

| | Geométrica | QNodes | Circuitos |
|---|---|---|---|
| Unidad de análisis | Arista | Subconjunto | Ciclo dirigido |
| Métrica de pérdida | EMD por corte de arista | EMD de distribución | Fuerza de circuitos rotos |
| Estructura que explota | Desconexión en grafo | Submodularidad de EMD | Base de ciclos del grafo |
| Tipo de espacio | Grafo dirigido | Espacio de probabilidad | Hipergrafo de circuitos |
| k-partición via | Dendrograma de cortes | Árbol de contracciones | Laplaciano espectral |

### Pseudocódigo

```
ALGORITHM CircuitStrategy(sistema, k):

// FASE 1: DESCUBRIR LA ESTRUCTURA DE CIRCUITOS
grafo = construir_grafo_causal(sistema)

// Encontrar todos los circuitos elementales del grafo dirigido
// (Johnson 1975 — O(E * (V+E)), el único algoritmo polinomial para esto)
circuitos = Johnson_TodosCircuitosElementales(grafo)

// Cada circuito tiene una "fuerza" — cuánta información causal transporta
// Un circuito fuerte = romperlo destruye mucha información
PARA cada circuito c en circuitos:
    fuerza(c) = producto_de_probabilidades_de_transición_a_lo_largo_de(c)

// FASE 2: CONSTRUIR EL HIPERGRAFO DE CIRCUITOS
// La partición ya no opera sobre el grafo original sino sobre este hipergrafo:
//   Nodos     = elementos del sistema
//   Hiperarcos = circuitos (cada circuito conecta sus nodos como un hiperarco)
//   Pesos     = fuerza de cada circuito
hipergrafo = (sistema.elementos, circuitos, fuerza)

// FASE 3: K-PARTICIÓN ESPECTRAL DEL HIPERGRAFO
// El Laplaciano del hipergrafo captura la estructura de ciclos del sistema.
// Sus eigenvectores revelan los grupos naturales: nodos en el mismo grupo
// comparten más ciclos fuertes entre sí que con el resto.

H   = matriz_incidencia(hipergrafo)         // |nodos| × |circuitos|
W   = diagonal(fuerza de cada circuito)
D_v = diagonal(suma de pesos de circuitos que contienen a cada nodo)

L_H = D_v - H * W * H^T    // Laplaciano generalizado del hipergrafo

// Los primeros k vectores propios de L_H embeben el sistema en R^k
// preservando la estructura de circuitos
[V, Λ] = k_vectores_propios_menores(L_H)

// Cada nodo queda representado como un punto en R^k.
// Nodos con comportamientos de ciclo similares están cerca.
// Agrupar por proximidad equivale a respetar los ciclos causales.
partición = k_medias(filas_de(V), k)

// FASE 4: CALCULAR PÉRDIDA REAL POR CIRCUITOS ROTOS
// Un circuito está "roto" si sus nodos pertenecen a más de un grupo.
Φ = 0
PARA cada circuito c en circuitos:
    SI los nodos de c pertenecen a más de un grupo en partición:
        Φ += fuerza(c)   // este ciclo causal está destruido por la partición

// FASE 5: REFINAMIENTO LOCAL (búsqueda local sobre EMD real)
// La partición espectral es una propuesta inicial, no necesariamente óptima.
// El refinamiento mueve un nodo a la vez y acepta el movimiento si reduce Φ.
mejoró = verdadero
MIENTRAS mejoró:
    mejoró = falso
    PARA cada nodo v:
        PARA cada grupo g ≠ grupo_actual(v):
            SI EMD(mover v a g) < EMD_actual:
                mover v a g
                mejoró = verdadero

RETORNAR partición, Φ
```

### El punto delicado: nodos puente

Un nodo que aparece en circuitos de múltiples grupos es un **nodo puente** — pertenece naturalmente a la frontera del sistema. La asignación óptima de estos nodos requiere el refinamiento local de la Fase 5. Una extensión futura podría usar un esquema de penalización suave en el Laplaciano para que los nodos puente reciban una señal más débil en la fase espectral y el refinamiento local tenga menos trabajo.

### Complejidad

| Fase | Operación | Costo |
|---|---|---|
| Encontrar circuitos | Johnson | O(E * (V+E)) |
| Construir Laplaciano | Multiplicación matricial | O(n² * |circuitos|) |
| Eigendescomposición | k eigenvectores de n×n | O(n³) |
| k-medias | k grupos, n puntos | O(k * n * iter) |
| Refinamiento local | Por cada nodo × por cada grupo | O(n * k * iter) |

El costo dominante es O(n³) por la eigendescomposición, igual que QNodes.

---

## Comparación de las tres estrategias

### ¿Cuándo usar cada una?

| Escenario | Estrategia recomendada | Razón |
|---|---|---|
| Sistema pequeño, k=2 | Geométrica | Dendrograma exacto, muy rápida |
| Sistema mediano, k≥3 | QNodes K | Árbol de contracciones, O(n³) exacta |
| Sistema con estructura cíclica conocida | Circuitos | Trabaja directamente con los ciclos causales |
| Sistema sin estructura previa | QNodes K | Garantías teóricas más fuertes |

### El problema abierto de las k-particiones

Las tres estrategias atacan el problema desde ángulos distintos, pero comparten un límite: ninguna garantiza el óptimo global para k > 2 en tiempo polinomial (el problema es NP-difícil para k ≥ 3 en general). La diferencia está en qué tan cerca del óptimo llegan y en qué tipo de sistemas funcionan mejor.

La estrategia de Circuitos abre una dirección nueva: en lugar de medir pérdida como EMD de distribuciones, medir pérdida como circuitos rotos. Si esta métrica resulta ser equivalente o aproximable por EMD, entonces el problema de k-particiones en IIT podría reducirse a hipergrafos, donde existen algoritmos de particionado con garantías de aproximación conocidas.

---

## Referencias

- **Dendrograma divisivo**: Ward (1963), clustering jerárquico divisivo
- **Queyranne (1998)**: algoritmo para minimizar funciones submodulares simétricas en O(n³)
- **Árbol de Gomory-Hu**: análogo para flujos de red, inspiración para el árbol de contracciones
- **Johnson (1975)**: todos los circuitos elementales de un grafo dirigido en O(E*(V+E))
- **Laplaciano de hipergrafo**: Zhou, Huang, Schölkopf (2006), aprendizaje sobre hipergrafos
- **IIT**: Tononi, Boly, Massimini, Koch (2016), integrated information theory 3.0
