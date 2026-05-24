# Manual de Usuario — K-QGMIP

**Proyecto:** ProyectoAnalisis2026  
**Repositorio:** https://github.com/CamiOso/BiparticionOptima  
**Autores:** CamiOso  
**Fecha:** Mayo 2026  
**Versión:** 1.0

---

## Tabla de Contenidos

1. [Introducción y Visión General](#1-introducción-y-visión-general)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Instalación Paso a Paso](#3-instalación-paso-a-paso)
4. [Video Tutorial](#4-video-tutorial)
5. [Guía de Uso Básico](#5-guía-de-uso-básico)
6. [Opciones y Parámetros Avanzados](#6-opciones-y-parámetros-avanzados)
7. [Solución de Problemas](#7-solución-de-problemas)
8. [Ejemplos y Tutoriales](#8-ejemplos-y-tutoriales)
9. [Referencia Rápida](#9-referencia-rápida)

---

## 1. Introducción y Visión General

### 1.1 ¿Qué hace este software?

Se usa este software para encontrar la mejor manera de dividir un sistema de nodos en grupos, de forma que se pierda la menor cantidad de información posible entre ellos. A ese proceso se le llama encontrar la **k-partición de mínima información** del sistema.

Un sistema de nodos es simplemente un conjunto de elementos que se influyen mutuamente a lo largo del tiempo. Por ejemplo, un grupo de neuronas donde cada una activa o desactiva a las demás. El software analiza cómo se comporta ese sistema y responde la pregunta: ¿cuál es la forma óptima de dividirlo en k grupos?

Se ofrecen dos estrategias principales para resolver este problema, que se denominan **KGeoMIP** y **KQNodes**. KGeoMIP es más rápida; KQNodes generalmente encuentra particiones de mejor calidad. El usuario elige cuál usar según sus necesidades de velocidad o precisión.

### 1.2 ¿Para qué sirve?

Se aplica este software principalmente en investigación relacionada con la Teoría de Información Integrada (IIT), que es un marco matemático para estudiar la integración de información en sistemas complejos, incluyendo sistemas neuronales.

Algunos casos de uso típicos son:

- Verificar si un sistema tiene estructura modular (es decir, si se puede dividir en partes relativamente independientes).
- Comparar la información integrada de un sistema para diferentes valores de k (número de grupos).
- Estudiar cómo cambia la pérdida de información al aumentar el número de grupos en la partición.
- Reproducir experimentos del paper de Tononi et al. sobre ejemplos de referencia.

### 1.3 ¿Qué es una k-partición y qué significa "mínima información"?

Una **k-partición** es simplemente una forma de organizar los nodos del sistema en k grupos. Por ejemplo, si el sistema tiene 4 nodos (0, 1, 2, 3) y k=2, una partición posible es: Grupo A = {0, 1} y Grupo B = {2, 3}.

La **partición de mínima información** es aquella en la que los grupos quedan lo más "independientes" posible entre sí. Se mide esa independencia con un número llamado **pérdida** (también llamado φ o phi): cuanto menor es ese número, más independientes son los grupos y más "natural" es ese corte del sistema.

En términos simples: el software busca el corte del sistema que más se parece a una división real, no forzada.

### 1.4 Capacidades y limitaciones

**El software puede:**

- Analizar sistemas de hasta 25 nodos, aunque los tiempos aumentan exponencialmente con el tamaño.
- Encontrar particiones en k = 2, 3, 4 o 5 grupos.
- Cargar la descripción del sistema desde archivos CSV incluidos en el proyecto o desde muestras externas.
- Guardar los resultados en formato JSON para análisis posterior.

**El software no puede:**

- Analizar sistemas de más de 25 nodos en tiempo razonable (para n > 20 los tiempos por fila pueden ser de varias horas).
- Garantizar el óptimo global para k > 2 y n > 8, ya que se usan heurísticas.
- Trabajar con variables continuas sin discretización previa; los nodos deben ser binarios (0 o 1).

> **Nota sobre tiempos:** Para sistemas pequeños (n ≤ 8) el análisis toma segundos. Para n = 15, entre 1 y 15 minutos. Para n = 20 con muchos nodos activos, puede tardar horas.

---

## 2. Requisitos del Sistema

### 2.1 Sistema operativo

Se soportan los siguientes sistemas operativos:

- **Linux:** Ubuntu 20.04 o superior, Debian 11 o superior, cualquier distribución con soporte para Python 3.11. **(Recomendado para experimentos grandes)**
- **macOS:** 12 Monterey o superior.
- **Windows:** Windows 10 (build 19041 o superior) o Windows 11, usando PowerShell o la terminal de Windows.

### 2.2 Hardware

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| Procesador | 2 núcleos, 2.0 GHz | 4+ núcleos, 3.0+ GHz |
| RAM | 4 GB | 8 GB para n ≤ 15; 16 GB para n ≤ 22; 32 GB para n = 25 |
| Espacio en disco | 500 MB | 5 GB (para archivos de resultados y TPMs grandes) |
| Conexión a internet | Solo para la descarga inicial | — |

> **Nota sobre RAM:** Las TPMs de sistemas grandes ocupan bastante espacio. Para n=25 la TPM ya ocupa ~3.4 GB en disco. Si la RAM disponible es menor a 16 GB se recomienda no intentar analizar sistemas con más de 20 nodos.

### 2.3 Software

Se requieren los siguientes programas y librerías:

| Software | Versión requerida | Para qué se usa |
|---------|-----------------|----------------|
| Python | 3.11 o superior | Lenguaje del proyecto |
| pip | Incluido con Python 3.11 | Instalar librerías |
| numpy | 2.2.5 | Cálculos vectoriales y matrices |
| pandas | 3.0.1 | Lectura y escritura de archivos de datos |
| openpyxl | 3.1.5 | Lectura del Excel de resultados |
| pytest | 9.0.2 | Ejecución de pruebas (opcional para usuarios) |
| Git | Cualquier versión reciente | Descargar el proyecto |

Todas las librerías principales se instalan automáticamente desde el archivo `requirements.txt` del proyecto. Las librerías de visualización (`matplotlib` y `networkx`) son opcionales y se necesitan solo si se quieren generar gráficas de las particiones:

```bash
pip install matplotlib networkx
```

---

## 3. Instalación Paso a Paso

### 3.1 Verificar que Python esté instalado

Se abre una terminal (en Windows: PowerShell o Símbolo del sistema; en Linux/macOS: Terminal) y se escribe:

```bash
python --version
```

Se debe ver una respuesta como `Python 3.11.x` o superior. Si aparece un error o una versión anterior a 3.11, se descarga Python desde [https://python.org/downloads](https://python.org/downloads) y se instala.

`[CAPTURA: Verificación de versión de Python en terminal]`

### 3.2 Verificar que Git esté instalado

En la misma terminal se escribe:

```bash
git --version
```

Se debe ver una respuesta como `git version 2.x.x`. Si Git no está instalado, se descarga desde [https://git-scm.com/downloads](https://git-scm.com/downloads).

### 3.3 Descargar el proyecto

Se descarga el proyecto desde GitHub con el siguiente comando:

```bash
git clone https://github.com/CamiOso/BiparticionOptima.git
```

Esto crea una carpeta llamada `BiparticionOptima` en el directorio actual. Se navega hasta esa carpeta:

```bash
cd BiparticionOptima
```

`[CAPTURA: Descarga exitosa del repositorio, mostrando el listado de archivos]`

### 3.4 Crear el entorno virtual

Se crea un entorno virtual de Python para que las librerías del proyecto no interfieran con otras instalaciones de Python en el sistema:

```bash
python -m venv .venv
```

Luego se activa el entorno virtual. El comando varía según el sistema operativo:

**En Linux o macOS:**
```bash
source .venv/bin/activate
```

**En Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**En Windows (Símbolo del sistema):**
```cmd
.venv\Scripts\activate.bat
```

Se sabe que el entorno está activo cuando el nombre `(.venv)` aparece al inicio de la línea de la terminal.

`[CAPTURA: Terminal mostrando el prefijo (.venv) indicando entorno activo]`

### 3.5 Instalar las librerías

Con el entorno virtual activo, se instalan todas las dependencias del proyecto con un solo comando:

```bash
pip install -r requirements.txt
```

Se espera a que termine la instalación, que suele tardar entre 1 y 3 minutos según la velocidad de la conexión a internet.

`[CAPTURA: Proceso de instalación de pip mostrando "Successfully installed"]`

### 3.6 Verificar que la instalación fue correcta

Se ejecuta el programa principal con el siguiente comando para comprobar que todo funciona:

```bash
python exec.py
```

Si la instalación fue correcta, se ve una salida similar a esta en la terminal:

```
ProyectoAnalisis2026 v1.0: proyecto iniciado correctamente con estrategia base FuerzaBruta.
Configuracion base -> distancia: hamming, notacion: lil_endian, tiempo EMD: emd_efecto.
TPM cargada desde src/.samples/N4A.csv con forma (16, 4).
Sistema demo -> distribucion marginal: [...]
FuerzaBruta -> ...
Perdida -> 0.xxxx | subsistema=[...] vs particion=[...]
...
```

`[CAPTURA: Salida completa del primer ejecutable exitoso]`

> **Importante:** Si aparece un error de tipo `ModuleNotFoundError`, se verifica que el entorno virtual esté activado (el prefijo `(.venv)` debe verse en la terminal antes de volver a ejecutar).

---

## 4. Video Tutorial

Se incluye un video tutorial que muestra el proceso completo de instalación y uso del software desde cero.

**Enlace al video:** `[AGREGAR ENLACE AL VIDEO AQUÍ]`

**Contenido del video:**
- Instalación completa del entorno desde un sistema limpio.
- Ejecución del primer análisis con el dataset de muestra N4A.
- Uso de diferentes valores de k (k=2, k=3) y comparación de resultados.
- Interpretación de la salida que produce el programa.
- Exportación de resultados a JSON.

**Duración:** aproximadamente 12 minutos.

Se recomienda ver el video antes de seguir las instrucciones escritas, ya que permite familiarizarse con la apariencia de la terminal y las respuestas esperadas del programa en cada paso.

---

## 5. Guía de Uso Básico

### 5.1 Estructura de un comando

Todos los análisis se ejecutan desde la raíz del proyecto (la carpeta `BiparticionOptima`) con el entorno virtual activo. El formato general de un comando es:

```bash
python exec.py --estrategia <nombre> --estado-inicial <bits> --k-particiones <k>
```

Los tres elementos que se pueden cambiar son:

- **`--estrategia`**: cuál de los dos algoritmos usar. Las opciones para k-particiones son `geometric` (KGeoMIP) y `qnodos` (KQNodes).
- **`--estado-inicial`**: el estado de los nodos en el momento t=0. Se escribe como una cadena binaria donde cada dígito corresponde a un nodo: `1` = nodo activo, `0` = nodo inactivo.
- **`--k-particiones`**: el número de grupos en que se divide el sistema (2, 3, 4 o 5).

### 5.2 Cómo se selecciona el dataset

El programa determina automáticamente qué dataset cargar **a partir de la longitud del `--estado-inicial`**. Si el estado tiene 4 caracteres, carga `N4A.csv`; si tiene 6, carga `N6A.csv`, y así sucesivamente. El sufijo `A` corresponde a la página de red por defecto del proyecto.

Por eso, para analizar un sistema de 6 nodos se debe pasar un estado de 6 dígitos:

```bash
python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 3
#                                                   ^^^^^^ 6 dígitos → carga N6A.csv
```

Los datasets disponibles y el estado sugerido para cada uno son:

| Archivo | Nodos | Tamaño TPM | Estado inicial (longitud) | Uso recomendado |
|---------|-------|-----------|--------------------------|----------------|
| `N4A.csv` | 4 | 16 × 4 | 4 dígitos, ej: `1000` | Validación y aprendizaje inicial |
| `N5A.csv` | 5 | 32 × 5 | 5 dígitos, ej: `10000` | Pruebas rápidas |
| `N6A.csv` | 6 | 64 × 6 | 6 dígitos, ej: `101010` | Pruebas medianas |
| `N7A.csv` | 7 | 128 × 7 | 7 dígitos, ej: `1000100` | Pruebas medianas |
| `N8A.csv` | 8 | 256 × 8 | 8 dígitos, ej: `10001000` | Benchmarks comparativos |
| `N10A.csv` | 10 | 1 024 × 10 | 10 dígitos | Experimentos intermedios |
| `N15B.csv` | 15 | 32 768 × 15 | 15 dígitos | Experimentos avanzados |
| `N20A.csv` | 20 | ~81 MB | 20 dígitos | Solo con 16 GB RAM+ |
| `N22A.csv` | 22 | ~353 MB | 22 dígitos | Solo con 16 GB RAM+ |

> Todos los archivos están en la carpeta `src/.samples/` dentro del proyecto. Si el archivo correspondiente al estado dado no existe, el programa muestra un mensaje de error con la lista de archivos disponibles.

### 5.3 Ejecutar el primer análisis

Se ejecuta el siguiente comando para analizar el sistema de 4 nodos con k=3 grupos usando KQNodes:

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3
```

`[CAPTURA: Terminal mostrando el comando y su ejecución]`

### 5.4 Leer e interpretar los resultados

Cuando el programa termina, imprime en pantalla un resultado similar al siguiente:

```
Q-Nodos ->
Estrategia   : qnodos_k3
Estado inicial: 1000
Perdida       : 0.0625
Particion     : Grupo 0 → mecanismo=(0,1) alcance=(0,1)
                Grupo 1 → mecanismo=(2,) alcance=(2,)
                Grupo 2 → mecanismo=(3,) alcance=(3,)
```

Cada parte de este resultado significa lo siguiente:

- **Estrategia:** confirma qué algoritmo se usó y con qué k.
- **Estado inicial:** el estado de los nodos con el que se hizo el análisis.
- **Pérdida:** el valor φ (phi) de la partición encontrada. Este es el dato más importante: indica qué tan "natural" es el corte. Un valor de 0.0 significa que los grupos son perfectamente independientes. Un valor alto indica que hay mucha integración entre los grupos.
- **Partición:** los grupos encontrados. Cada grupo muestra qué nodos quedaron juntos en el mecanismo (tiempo presente, t=0) y en el alcance (tiempo futuro, t=1).

`[CAPTURA: Salida completa con anotaciones señalando cada campo del resultado]`

### 5.5 Casos de uso típicos

**Comparar KGeoMIP y KQNodes para el mismo sistema:**

```bash
python exec.py --estrategia geometric --estado-inicial 1000 --k-particiones 3
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3
```

Se comparan las pérdidas que devuelven los dos comandos. Si KQNodes da un valor menor, encontró una partición de mejor calidad.

**Analizar el mismo sistema para diferentes valores de k:**

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 2
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 4
```

Se espera que la pérdida disminuya o se mantenga igual a medida que k aumenta, ya que más grupos permiten cortes más naturales.

**Ejecutar todas las estrategias a la vez (modo comparativo):**

```bash
python exec.py --estado-inicial 1000
```

Cuando no se indica `--estrategia`, el programa ejecuta automáticamente FuerzaBruta, Phi, QNodos y Geometric en secuencia y muestra todos los resultados. Se usa este modo solo para sistemas pequeños (n ≤ 6), ya que FuerzaBruta es muy lenta para sistemas más grandes.

---

## 6. Opciones y Parámetros Avanzados

### 6.1 Lista completa de parámetros

| Parámetro | Valores posibles | Valor por defecto | Descripción |
|-----------|-----------------|------------------|-------------|
| `--estrategia` | `todas`, `fuerza_bruta`, `phi`, `qnodos`, `geometric` | `todas` | Algoritmo a usar |
| `--estado-inicial` | Cadena binaria, ej: `1000` | `1000` | Estado de los nodos en t=0 |
| `--k-particiones` | Entero ≥ 2 | `2` | Número de grupos en la partición |
| `--modo-geometric` | `refinado`, `estricto` | `refinado` | Modo de KGeoMIP (solo aplica con `--estrategia geometric`) |
| `--output-json` | Ruta de archivo, ej: `resultado.json` | No se guarda | Exportar resultado en formato JSON |
| `--csv-muestras` | Ruta de archivo CSV | No se usa | Estimar la TPM desde datos propios |

### 6.2 El modo de KGeoMIP: `refinado` vs `estricto`

El parámetro `--modo-geometric` solo aplica cuando se usa `--estrategia geometric`. Se elige entre:

- **`refinado`** (por defecto): aplica búsqueda con hill-climbing y reinicios aleatorios además de la búsqueda geométrica base. Se elige este modo ya que da resultados de mayor calidad. Es el modo recomendado para la mayoría de los casos.
- **`estricto`**: aplica solo la búsqueda geométrica base, sin refinamientos adicionales. Es más rápido pero puede encontrar particiones ligeramente peores.

Ejemplo:
```bash
python exec.py --estrategia geometric --modo-geometric estricto --k-particiones 3
```

### 6.3 Exportar resultados a JSON

Se agrega `--output-json` para guardar los resultados en un archivo. Esto es útil cuando se quiere analizar los resultados con otro programa o guardar un registro de los experimentos:

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3 \
               --output-json resultados/mi_analisis.json
```

El archivo JSON tiene la siguiente estructura:

```json
{
  "estrategia_solicitada": "qnodos",
  "modo_geometric": "refinado",
  "k_particiones": 3,
  "estado_inicial": "1000",
  "archivo_tpm": "src/.samples/N4A.csv",
  "resultados": {
    "qnodos_k3": {
      "estrategia": "qnodos_k3",
      "estado_inicial": "1000",
      "perdida": 0.0625,
      "particion": "...",
      "distribucion_subsistema": [...],
      "distribucion_particion": [...],
      "elapsed_seconds": 0.38
    }
  }
}
```

Los campos más importantes son `perdida` (el valor φ encontrado) y `elapsed_seconds` (cuántos segundos tardó).

### 6.4 Usar datos propios con `--csv-muestras`

Si se tiene una serie temporal de observaciones de un sistema propio, se puede estimar la TPM a partir de esas observaciones y luego analizarla. El CSV debe tener el siguiente formato:

- Cada fila es un instante de tiempo.
- Cada columna es un nodo.
- Los valores son `0` o `1` (nodo inactivo o activo).

Ejemplo de archivo `mis_datos.csv`:

```
1,0,1,0
0,1,0,1
1,1,0,0
0,0,1,1
1,0,0,1
```

Se ejecuta con:

```bash
python exec.py --estrategia qnodos --estado-inicial 1010 \
               --csv-muestras mis_datos.csv --k-particiones 3
```

El programa estima la TPM a partir de las frecuencias observadas en el CSV y luego aplica el análisis normalmente.

### 6.5 Visualización gráfica de particiones

Se generan gráficas de los resultados usando el módulo `src/visualizacion/particion.py`. Se necesitan `matplotlib` y `networkx` instalados (ver sección 2.3). Estas funciones se llaman desde un script Python, no desde la línea de comandos.

**Graficar una bipartición (k=2):**

```python
from src.visualizacion.particion import dibujar_biparticion

dibujar_biparticion(
    subalcance=(0, 1),           # nodos del Grupo A en el alcance futuro
    submecanismo=(0,),           # nodos del Grupo A en el mecanismo presente
    alcance_total=(0, 1, 2, 3),  # todos los nodos del alcance
    mecanismo_total=(0, 1, 2, 3), # todos los nodos del mecanismo
    perdida=0.0625,
    guardar_en="resultados/biparticion_n4a.png",  # omitir para mostrar en pantalla
)
```

Esto genera una imagen donde los nodos azules pertenecen al Grupo A, los naranja al Grupo B, las aristas sólidas son las conexiones que la partición conserva y las punteadas son las que corta.

`[CAPTURA: Ejemplo de gráfica de bipartición generada]`

**Graficar una k-partición (k=3 o más):**

```python
from src.visualizacion.particion import dibujar_k_particion

dibujar_k_particion(
    nodos=[0, 1, 2, 3],
    asignacion=(0, 0, 1, 2),     # cada nodo a qué grupo pertenece
    alcance_total=(0, 1, 2, 3),
    mecanismo_total=(0, 1, 2, 3),
    perdida=0.0312,
    guardar_en="resultados/k_particion_n4a_k3.png",
)
```

Cada grupo recibe un color diferente. Las aristas sólidas conectan nodos del mismo grupo y las punteadas conectan nodos de grupos distintos.

`[CAPTURA: Ejemplo de gráfica de k-partición con 3 grupos coloreados]`

**Comparar pérdidas de múltiples estrategias en una gráfica de barras:**

```python
from src.visualizacion.particion import dibujar_comparacion_perdidas

dibujar_comparacion_perdidas(
    {
        "KGeoMIP k=2": 0.125,
        "KQNodes k=2": 0.125,
        "KGeoMIP k=3": 0.0625,
        "KQNodes k=3": 0.0312,
    },
    guardar_en="resultados/comparacion.png",
)
```

`[CAPTURA: Gráfica de barras comparando pérdidas entre estrategias]`

### 6.6 Archivo de resultados Excel

Los experimentos a gran escala (N20A, N22A, N25A) guardan sus resultados en el archivo `DatosPruebas2026_1.xlsx`, que está en la raíz del proyecto. Se abre con Excel, LibreOffice Calc o cualquier lector de hojas de cálculo.

El archivo tiene tres hojas principales:

| Hoja | Contenido |
|------|-----------|
| `20A-Elementos` | Resultados de experimentos con el sistema N20A (20 nodos) |
| `22A-Elementos` | Resultados de experimentos con el sistema N22A (22 nodos) |
| `25A-Elementos` | Resultados de experimentos con el sistema N25A (25 nodos) |

Cada hoja tiene columnas para la partición encontrada, la pérdida φ y el tiempo de ejecución, separadas por estrategia (KQNodes y KGeoMIP) y por valor de k (k=2, 3, 4, 5). Las celdas vacías indican que ese experimento aún no se completó o fue excluido por restricción de tiempo.

> No se recomienda editar este archivo manualmente ya que los scripts de experimentos lo actualizan automáticamente con bloqueo de archivo para evitar corrupción.

### 6.7 Consejos de rendimiento

- Se recomienda usar KGeoMIP (`geometric`) cuando n ≤ 12 ya que aplica una búsqueda exacta por DP que es muy rápida.
- Se recomienda usar KQNodes (`qnodos`) cuando n > 12 ya que su semilla de árbol de contracciones es de mayor calidad para sistemas grandes.
- Para n > 15 se aconseja cerrar otros programas pesados antes de ejecutar el análisis, ya que el consumo de memoria puede ser significativo.
- Si el análisis tarda demasiado, se puede reducir k o usar un sistema más pequeño para hacer una prueba primero.
- Para experimentos muy largos (n ≥ 20) se usan los scripts especializados de la carpeta `scripts/` descritos en la sección 8.4.

---

## 7. Solución de Problemas

### 7.1 Errores durante la instalación

**Error: `pip: command not found`**

Se instala pip siguiendo las instrucciones oficiales: [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/). En la mayoría de los casos basta con ejecutar `python -m ensurepip --upgrade`.

**Error: `python: command not found` o versión incorrecta**

En algunos sistemas la versión 3.11 está disponible como `python3` o `python3.11`. Se intenta:
```bash
python3 --version
python3.11 --version
```
Si se usa `python3`, todos los comandos del manual se deben adaptar reemplazando `python` por `python3`.

**Error al activar el entorno virtual en Windows: "No se puede cargar el archivo porque la ejecución de scripts está deshabilitada"**

Se abre PowerShell como administrador y se ejecuta:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego se cierra PowerShell y se vuelve a abrir normalmente.

**Error: `ERROR: Could not find a version that satisfies the requirement numpy==2.2.5`**

Esto indica que pip está desactualizado. Se actualiza con:
```bash
pip install --upgrade pip
```
Y luego se intenta de nuevo `pip install -r requirements.txt`.

### 7.2 Errores durante la ejecución

**Error: `ModuleNotFoundError: No module named 'numpy'`**

El entorno virtual no está activo. Se verifica que el prefijo `(.venv)` aparece en la terminal. Si no aparece, se activa de nuevo:
```bash
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate.bat    # Windows
```

**Error: `ValueError: Estado inicial debe ser una cadena binaria`**

El `--estado-inicial` contiene caracteres que no son `0` o `1`. Se verifica que la cadena esté formada exclusivamente por ceros y unos, sin espacios ni otras letras.

**Error: `ValueError: k_particiones debe ser >= 2`**

Se verificó que el valor de `--k-particiones` sea un número entero mayor o igual a 2.

**Error: `FileNotFoundError: archivo_tpm no encontrado`**

El programa no encuentra el dataset. Esto ocurre cuando se ejecuta `exec.py` desde una carpeta diferente a la raíz del proyecto. Se verifica que el directorio de trabajo sea `BiparticionOptima`:
```bash
pwd          # Linux/macOS — debe mostrar la ruta hasta BiparticionOptima
cd           # Windows — muestra el directorio actual
```

**El programa tarda demasiado y no termina**

Esto es esperado para sistemas grandes con muchos nodos activos. Para n=20 con 18-20 nodos activos, el análisis puede tardar entre 4 y 12 horas. Si no se dispone de ese tiempo, se recomienda interrumpir con `Ctrl+C` y probar con un sistema más pequeño (N10A o N8A).

**El programa consume demasiada memoria y el sistema se vuelve lento**

Esto ocurre especialmente con los datasets N20A, N22A y N25A. Se cierra el programa con `Ctrl+C` y se intenta con un dataset más pequeño. Para usar N20A se necesitan al menos 8 GB de RAM libre.

### 7.3 Resultados inesperados

**La pérdida es 0.0**

Un valor de pérdida igual a cero indica que los grupos son perfectamente independientes. Esto es un resultado válido; significa que el sistema tiene una estructura modular clara y no hay información integrada entre los grupos encontrados.

**KGeoMIP y KQNodes dan pérdidas muy diferentes**

Esto es normal. KQNodes generalmente encuentra particiones de menor pérdida (mejor calidad) a costa de ser más lento. Si la diferencia es grande, se confía más en el resultado de KQNodes.

**El resultado cambia cada vez que se ejecuta**

Algunas partes del algoritmo usan recocido simulado, que tiene un componente aleatorio. Sin embargo, la semilla está fijada en el código (`semilla_numpy = 73`) por lo que los resultados deben ser reproducibles. Si cambian, puede indicar un problema con la configuración del entorno.

---

## 8. Ejemplos y Tutoriales

### 8.1 Tutorial básico — Sistema de 4 nodos (N4A)

Este tutorial cubre el análisis completo de un sistema pequeño, desde la preparación hasta la interpretación de resultados.

**Paso 1: Verificar que el entorno esté activo**

Se abre la terminal, se navega a la carpeta `BiparticionOptima` y se activa el entorno virtual:

```bash
cd BiparticionOptima
source .venv/bin/activate     # Linux/macOS
```

Se verifica que el prefijo `(.venv)` aparece.

**Paso 2: Ejecutar la bipartición de referencia (k=2)**

Se comienza con k=2 para tener un punto de comparación:

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 2
```

Se toma nota de la pérdida que aparece en la salida.

`[CAPTURA: Salida del comando con k=2]`

**Paso 3: Ejecutar con k=3 y comparar**

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3
```

Se compara la pérdida con la del paso anterior. Se espera que sea menor o igual, ya que con más grupos el sistema tiene más flexibilidad para encontrar un corte natural.

`[CAPTURA: Salida del comando con k=3 mostrando la pérdida y la partición]`

**Paso 4: Repetir con KGeoMIP y comparar velocidades**

```bash
python exec.py --estrategia geometric --estado-inicial 1000 --k-particiones 3
```

Se compara la pérdida encontrada por KGeoMIP con la de KQNodes. Se observa también la diferencia en tiempo de ejecución.

**Paso 5: Guardar los resultados**

```bash
python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3 \
               --output-json resultados/tutorial_n4a.json
```

Se abre el archivo `resultados/tutorial_n4a.json` con cualquier editor de texto para ver el resultado completo en formato estructurado.

`[CAPTURA: Archivo JSON abierto en editor de texto mostrando la estructura]`

### 8.2 Caso de estudio intermedio — Sistema de 6 nodos con diferentes estados iniciales

Se analiza cómo cambia la partición óptima según el estado inicial del sistema.

**Análisis para tres estados diferentes con k=3:**

```bash
python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 3 \
               --output-json resultados/n6a_101010.json

python exec.py --estrategia qnodos --estado-inicial 111000 --k-particiones 3 \
               --output-json resultados/n6a_111000.json

python exec.py --estrategia qnodos --estado-inicial 100001 --k-particiones 3 \
               --output-json resultados/n6a_100001.json
```

`[CAPTURA: Los tres comandos ejecutándose en secuencia]`

Se comparan las pérdidas de los tres análisis. El estado con menor pérdida indica en qué configuración el sistema está más integrado. El estado con mayor pérdida es en el que el sistema es más fácilmente divisible.

**Comparar k=2 vs k=3 vs k=4 para el mismo estado:**

```bash
python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 2
python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 3
python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 4
```

Se construye una tabla con las pérdidas:

| k | Pérdida encontrada |
|---|------------------|
| 2 | (resultado k=2) |
| 3 | (resultado k=3) |
| 4 | (resultado k=4) |

Si la pérdida baja mucho al pasar de k=2 a k=3 pero poco al pasar de k=3 a k=4, el sistema tiene naturalmente 3 módulos.

### 8.3 Ejemplo avanzado — Análisis de sistema de 8 nodos con exportación

Se analiza un sistema de 8 nodos (N8A) con optimización de rendimiento y comparación completa de estrategias.

**Paso 1: Ejecutar KQNodes con k=3 y guardar resultado**

```bash
python exec.py --estrategia qnodos --estado-inicial 10001000 --k-particiones 3 \
               --output-json resultados/n8a_qnodos_k3.json
```

**Paso 2: Ejecutar KGeoMIP en modo refinado con el mismo sistema**

```bash
python exec.py --estrategia geometric --modo-geometric refinado \
               --estado-inicial 10001000 --k-particiones 3 \
               --output-json resultados/n8a_geo_k3.json
```

`[CAPTURA: Ambas salidas comparando tiempos y pérdidas]`

**Paso 3: Comparar los archivos JSON**

Se abren los dos archivos JSON y se comparan los campos `perdida` y `elapsed_seconds`. KGeoMIP debería ser significativamente más rápido, y KQNodes debería tener una pérdida menor o igual.

**Paso 4: Probar con datos propios usando `--csv-muestras`**

Si se tiene una serie temporal propia de observaciones binarias, se puede estimar la TPM:

```bash
python exec.py --estrategia qnodos --estado-inicial 10001000 \
               --csv-muestras mis_observaciones.csv --k-particiones 3
```

`[CAPTURA: Salida mostrando "TPM estimada desde muestras temporales" en lugar de la carga desde CSV]`

### 8.4 Experimentos a gran escala — Sistemas N20A, N22A, N25A

Para sistemas de 20, 22 y 25 nodos se usan scripts especializados en la carpeta `scripts/` ya que cada fila de experimento puede tardar horas. Estos scripts ejecutan un análisis por fila del Excel, guardan el resultado y terminan, lo que permite lanzar varias instancias en paralelo sin que interfieran.

> **Advertencia:** Este nivel de uso requiere al menos 16 GB de RAM y, para N25A, idealmente 32 GB. Se recomienda hacerlo solo en el computador de trabajo con todos los demás programas cerrados.

**Ejecutar una fila de experimento para N20A con KQNodes:**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -u scripts/run_qnodos_single.py 10
```

El número `10` es el índice de fila del Excel (hoja `20A-Elementos`). Las variables `OMP_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1` son importantes: evitan que numpy cree demasiados hilos internos que compiten entre sí cuando hay varias instancias corriendo en paralelo.

**Reanudar un experimento interrumpido desde un k específico:**

```bash
python3 -u scripts/run_geo_single.py 11 --start-k 3
```

Esto reanuda el experimento de la fila 11 comenzando desde k=3 en lugar de k=2, útil cuando el proceso fue interrumpido a mitad.

**Lanzar la cola completa de experimentos en segundo plano:**

```bash
nohup bash run_qnodos_cola.sh > /tmp/qnodos_log.log 2>&1 &
```

La cola encadena automáticamente N20A → N22A → N25A. Se usa `nohup` para que el proceso continúe aunque se cierre la terminal. El archivo `/tmp/qnodos_log.log` acumula toda la salida.

**Scripts disponibles:**

| Script | Estrategia | Sistema |
|--------|-----------|---------|
| `scripts/run_qnodos_single.py` | KQNodes | N20A |
| `scripts/run_qnodos_single_22A.py` | KQNodes | N22A |
| `scripts/run_qnodos_single_25A.py` | KQNodes | N25A |
| `scripts/run_geo_single.py` | KGeoMIP | N20A |
| `scripts/run_geo_single_22A.py` | KGeoMIP | N22A |
| `scripts/run_geo_single_25A.py` | KGeoMIP | N25A |
| `run_qnodos_cola.sh` | KQNodes | N20A → N22A → N25A |
| `run_geo_cola.sh` | KGeoMIP | N20A → N22A → N25A |

Los resultados se guardan automáticamente en `DatosPruebas2026_1.xlsx` en la hoja correspondiente al sistema analizado.

`[CAPTURA: Terminal mostrando la ejecución de run_qnodos_single.py con el progreso imprimiéndose por fila]`

---

## 9. Referencia Rápida

### 9.1 Comandos principales

| Tarea | Comando |
|------|---------|
| Primera ejecución (N4A, todas las estrategias) | `python exec.py` |
| KQNodes, k=3, sistema N4A | `python exec.py --estrategia qnodos --estado-inicial 1000 --k-particiones 3` |
| KQNodes, k=3, sistema N6A | `python exec.py --estrategia qnodos --estado-inicial 101010 --k-particiones 3` |
| KGeoMIP, k=3, modo refinado | `python exec.py --estrategia geometric --modo-geometric refinado --k-particiones 3` |
| KGeoMIP, modo estricto | `python exec.py --estrategia geometric --modo-geometric estricto` |
| Guardar resultado en JSON | `python exec.py --estrategia qnodos --output-json resultado.json` |
| Usar datos propios (CSV) | `python exec.py --estrategia qnodos --csv-muestras mis_datos.csv` |
| Solo fuerza bruta (exacto, n≤8) | `python exec.py --estrategia fuerza_bruta` |
| Instalar librerías de visualización | `pip install matplotlib networkx` |
| Experimento N20A fila 10 (KQNodes) | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u scripts/run_qnodos_single.py 10` |
| Ejecutar pruebas del proyecto | `PYTHONPATH=. python -m pytest -q` |

### 9.2 Tabla de parámetros

| Parámetro | Tipo | Por defecto | Valores válidos |
|-----------|------|------------|----------------|
| `--estrategia` | texto | `todas` | `todas`, `fuerza_bruta`, `phi`, `qnodos`, `geometric` |
| `--estado-inicial` | cadena binaria | `1000` | Cualquier cadena de 0s y 1s (ej: `101010`) |
| `--k-particiones` | entero | `2` | `2`, `3`, `4`, `5` (se recomiendan ≤ 5 para tiempos razonables) |
| `--modo-geometric` | texto | `refinado` | `refinado`, `estricto` |
| `--output-json` | ruta de archivo | (no guarda) | Cualquier ruta de archivo, ej: `resultados/mi_analisis.json` |
| `--csv-muestras` | ruta de archivo | (no usa) | Ruta a un CSV binario (filas=tiempos, columnas=nodos) |

### 9.3 Datasets de muestra disponibles

| Dataset | Nodos | Estado inicial sugerido | Tiempo esperado (k=3, KQNodes) |
|---------|-------|------------------------|-------------------------------|
| `N4A.csv` | 4 | `1000` | < 1 segundo |
| `N5A.csv` | 5 | `10000` | < 2 segundos |
| `N6A.csv` | 6 | `101010` | < 5 segundos |
| `N7A.csv` | 7 | `1000100` | < 15 segundos |
| `N8A.csv` | 8 | `10001000` | < 60 segundos |
| `N10A.csv` | 10 | `1000000000` | 1–5 minutos |
| `N15B.csv` | 15 | `100000000000000` | 5–30 minutos |

> Los archivos se cargan automáticamente según el tamaño del `--estado-inicial`. Un estado de 4 bits carga N4A; uno de 6 bits carga N6A, y así sucesivamente.

### 9.4 Glosario

**k-partición:** Manera de dividir los nodos de un sistema en k grupos. Por ejemplo, k=3 significa 3 grupos.

**Pérdida (φ / phi):** Número que mide cuánta información se pierde al cortar el sistema con esa partición. Un valor de 0 indica corte perfecto (grupos totalmente independientes). Valores más altos indican más integración.

**MIP (Minimum Information Partition):** La k-partición que tiene la menor pérdida de información posible. Es el resultado que busca el software.

**KGeoMIP:** Estrategia del proyecto que usa la geometría del sistema para encontrar la MIP. Es más rápida pero puede ser menos precisa.

**KQNodes:** Estrategia del proyecto que usa el algoritmo de Queyranne para encontrar la MIP. Es más lenta pero generalmente encuentra particiones de mejor calidad.

**Estado inicial:** Los valores de los nodos en un momento específico (t=0). Se especifica como cadena binaria donde `1` = nodo activo y `0` = nodo inactivo.

**TPM (Transition Probability Matrix):** Matriz de probabilidades de transición. Describe cómo evoluciona el sistema de un estado al siguiente. El software la carga desde los archivos CSV de muestra.

**Entorno virtual (.venv):** Carpeta aislada donde se instalan las librerías del proyecto sin afectar otras instalaciones de Python en el sistema. Es necesario activarlo antes de ejecutar el programa.

**Recocido simulado (SA):** Técnica de optimización que explora el espacio de soluciones de manera aleatoria controlada. Ambas estrategias lo usan internamente para refinar la partición encontrada.

---

*Fin del Manual de Usuario K-QGMIP — ProyectoAnalisis2026*
