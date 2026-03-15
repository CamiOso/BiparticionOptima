# Nota tecnica: complejidad de Geometric

## Objetivo

La estrategia `Geometric` se separa en dos modos para no mezclar una afirmacion teorica con optimizaciones practicas.

- `estricto`: modo orientado a justificar la cota teorica.
- `refinado`: modo orientado a maximizar precision empirica frente a `FuerzaBruta`.

## Modo estricto

El modo `estricto` construye una busqueda sobre mascaras del hipercubo de dimension `n`.

### Esquema

1. Se recorre cada mascara del hipercubo una vez.
2. Para cada mascara se calcula un costo local.
3. La tabla recursiva acumula costos usando transiciones de un bit.
4. Se selecciona un conjunto reducido de mascaras base.
5. Sobre esas mascaras se evalua una vecindad geometrica acotada.

### Lectura de complejidad

El espacio de estados tiene tamano `2^n`.

- Recorrido de mascaras: `O(2^n)`.
- Por cada mascara se exploran hasta `n` transiciones elementales: `O(n)`.
- La tabla recursiva base queda entonces en `O(n · 2^n)`.

Bajo esta lectura, la parte geometricamente estructurada del algoritmo se presenta como:

$$
T_{estricto}(n) = O(n \cdot 2^n)
$$

Esa es la cota que se defiende teoricamente para el modo base.

## Modo refinado

El modo `refinado` toma la salida del modo estricto y agrega:

- refinamiento local desacoplado,
- expansion adaptativa cuando detecta incertidumbre,
- restarts deterministas en casos grandes.

Estas fases mejoran la precision empirica, pero agregan trabajo dependiente de:

- cantidad de semillas refinadas,
- radio de vecindad explorado,
- numero de iteraciones de hill-climbing,
- numero de restarts.

Por eso, el modo `refinado` ya no debe presentarse como una implementacion estrictamente `O(n · 2^n)` en todos los casos.

## Interpretacion recomendada

- Si el objetivo es justificar la cota teorica: usar `Geometric` en modo `estricto`.
- Si el objetivo es obtener mejor aproximacion a `FuerzaBruta`: usar `Geometric` en modo `refinado`.

## Conclusiones practicas

En este proyecto:

- `estricto` sirve para sostener la narrativa teorica del algoritmo geometrico.
- `refinado` sirve para la comparacion experimental y la calidad de resultados.
- ambos modos comparten la misma base geometricamente estructurada.
