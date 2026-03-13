# ProyectoAnalisis2026

Proyecto para construir, paso a paso, la logica de busqueda de biparticion optima.
## 1. Requisitos

- Python 3.11 o superior
- `pip`
- Git (opcional, para flujo de commits)

## 2. Clonar y entrar al proyecto

```bash
git clone https://github.com/CamiOso/BiparticionOptima.git
cd BiparticionOptima
```

Si ya estas en el workspace local, entra a:

```bash
cd /home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026
```

## 3. Crear y activar entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Nota:
- `requirements.txt` contiene lo minimo para correr pruebas (`pytest`).
- Si quieres forzar modo `PyPhi` en la estrategia `Phi`, instala `pyphi` manualmente.

## 5. Ejecutar el proyecto

```bash
python exec.py
```

Esto ejecuta `src/main.py` y muestra una demo de:
- `FuerzaBruta`
- `Phi` (PyPhi real si esta disponible, o heuristica)
- `QNodos` (version submodular con memoizacion)

## 6. Correr pruebas

```bash
python -m pytest -q
```

## 7. Estructura principal

```text
src/
	constantes/      # Mensajes, etiquetas y configuracion base
	controladores/   # Carga de TPMs (CSV de muestras)
	funciones/       # Utilidades IIT, particiones y formato
	intermedios/     # Logging y perfilado
	modelos/         # Aplicacion, sistema, ncubo, solucion
	estrategias/     # FuerzaBruta, Phi, QNodos
	main.py          # Orquestador de ejecucion
exec.py            # Entry point
tests/             # Suite de pruebas
```

## 8. Flujo recomendado de trabajo

1. Crear/activar entorno virtual.
2. Instalar dependencias.
3. Ejecutar `python exec.py` para validacion rapida.
4. Ejecutar `python -m pytest -q` antes de cada commit.
5. Hacer cambios pequenos, validar, y luego commit/push.

## 9. Estado actual

- Carpeta y modulos en espanol.
- Estrategias funcionales con pruebas automatizadas.
- `SIA` ya aplica `condicion`, `alcance` y `mecanismo` al preparar subsistema.
- `QNodos` ya usa una logica submodular con memoizacion.
- `Phi` usa `PyPhi` cuando esta disponible; si no, usa ruta heuristica.

## 10. Siguiente objetivo

Llevar a paridad completa con `Proyecto-2025A`, especialmente:
- analisis completo de red en `FuerzaBruta` (candidatos/subsistemas/reporte),
- paridad avanzada de `Phi` (causa/efecto y repertorios),
- utilidades IIT restantes para equivalencia total.
