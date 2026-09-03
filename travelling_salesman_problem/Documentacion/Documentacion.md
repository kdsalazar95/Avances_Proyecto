# Solución del TSP con Dijkstra

## Idea general

Dijkstra encuentra el camino más corto entre dos **nodos** de un grafo. El
Problema del Viajante (TSP) pide algo distinto: una ruta que pase por
**todas** las ciudades exactamente una vez y regrese al origen. Dijkstra
por sí solo no distingue si ya visitó una ciudad o no, así que no puede
resolver el TSP aplicado directamente sobre el grafo de ciudades.

La solución: en vez de correr Dijkstra sobre el grafo de ciudades, se
corre sobre un **grafo de estados**, donde cada "nodo" es en realidad el
par:

```
estado = (ciudad_actual, mascara_de_ciudades_visitadas)
```

La `mascara` es un número binario donde el bit *i* indica si la ciudad
*i* ya fue visitada (ej. `0b00101` = se visitaron las ciudades 0 y 2).
Con esta representación, Dijkstra vuelve a ser aplicable de forma exacta:
cada estado es un nodo distinto, y avanzar de un estado a otro (visitar
una ciudad nueva) es una arista con costo = la distancia entre ciudades.

Esto es equivalente al algoritmo de **Held-Karp**, implementado con la
mecánica de Dijkstra (cola de prioridad) en vez de programación dinámica
clásica. Es exacto (garantiza el óptimo), pero el número de estados
crece como `2^n * n`, así que solo es práctico para pocas ciudades
(hasta ~15-18).

## Estructura de archivos

```
comun/grafo.py            → representación pura del mapa de ciudades
Dijkstra/problema_tsp.py  → construye el grafo de ESTADOS a partir del grafo de ciudades
Dijkstra/dijkstra.py      → algoritmo de Dijkstra genérico (no sabe nada de TSP)
Dijkstra/reg_busqueda.py  → registra cada estado expandido durante la búsqueda
Dijkstra/main.py          → arma todo, ejecuta y muestra el resultado
```

La idea de diseño es separar **la representación del problema** (qué es
un estado, cómo se llega de uno a otro) del **algoritmo que lo resuelve**
(cómo explorar esos estados en el orden correcto). Esto permite que
`dijkstra.py` sea reutilizable para cualquier otro problema de camino
mínimo, no solo el TSP.

---
## Descripción de scripts

## `comun/grafo.py`

Estructura de datos pura: nodos (ciudades) + aristas con peso (distancia).
No sabe nada de TSP ni de estados — es solo el mapa.

| Función | Qué hace |
|---|---|
| `agregar_ciudad(nombre)` | Agrega una ciudad si no existe, y le asigna un índice numérico consecutivo. Si ya existe, devuelve su índice sin duplicarla. |
| `agregar_arista(ciudad_a, ciudad_b, peso)` | Agrega ambas ciudades (si no existen) y guarda la distancia entre ellas. El grafo es **no dirigido**: la arista `(A,B)` es la misma que `(B,A)`. |
| `peso(ciudad_a, ciudad_b)` | Devuelve la distancia entre dos ciudades. Acepta nombres o índices indistintamente. |
| `vecinos_de(ciudad)` | Devuelve `[(ciudad_vecina, peso), ...]` para todas las ciudades conectadas directamente. |
| `nodos` (propiedad) | Lista de nombres de ciudades, en orden de índice. |
| `n` (propiedad) | Cantidad de ciudades en el grafo. |
| `indice_de(nombre)` / `nombre_de(indice)` | Traducen entre el nombre de una ciudad y su índice numérico interno. |

---

### `problema_tsp.py`

Toma un `Grafo` de ciudades y lo convierte en el grafo de **estados**
`(ciudad, mascara)` que Dijkstra necesita.

| Función | Qué hace |
|---|---|
| `__init__(grafo, ciudad_inicio)` | Guarda el grafo, calcula cuántas ciudades hay (`n`) y prepara `mascara_completa` (todos los bits en 1, es decir, todas las ciudades visitadas). |
| `estado_inicial()` | Devuelve `(ciudad_inicio, mascara_con_solo_esa_ciudad)` — el punto de partida antes de moverse. |
| `es_meta(estado)` | `True` si la máscara del estado ya tiene **todas** las ciudades visitadas (aún sin cerrar el ciclo de regreso). |
| `vecinos(estado)` | Para el estado `(ciudad, mascara)`, devuelve todos los estados `(ciudad_nueva, mascara_actualizada)` alcanzables moviéndose a una ciudad **no visitada**, junto con el costo (la distancia entre ambas ciudades, tomada del `Grafo`). Las ciudades ya visitadas quedan excluidas por construcción. |
| `costo_cierre_ciclo(ciudad)` | Costo de volver de `ciudad` al punto de partida — se usa al final, para cerrar el recorrido. |
| `nombres_de_ruta(ruta_indices)` | Traduce una lista de índices de ciudades a sus nombres legibles. |

---

### `dijkstra.py`

El algoritmo, completamente genérico: no sabe qué es una ciudad. Solo
necesita que el `problema` que recibe tenga `estado_inicial()` y
`vecinos(estado)`.

| Función | Qué hace |
|---|---|
| `dijkstra(problema, registro=None)` | Ejecuta el algoritmo clásico: mantiene una cola de prioridad (`heapq`) con `(costo_acumulado, estado)`, siempre expandiendo primero el estado de menor costo conocido. Por cada estado que expande, revisa sus vecinos y actualiza el costo si encontró un camino más barato hacia ellos. Si se le pasa un `registro`, guarda una fila por cada estado que expande (no cuenta los que descarta por ya estar visitados). Devuelve dos diccionarios: `costos` (menor costo conocido para llegar a cada estado) y `padres` (de qué estado se llegó a cada uno, para poder reconstruir la ruta). |
| `reconstruir_camino(padres, estado_final, estado_inicial)` | Sigue la cadena de `padres` hacia atrás, desde el estado final hasta el inicial, y devuelve la secuencia completa de estados en el orden correcto (ida, no vuelta). |

**Por qué es exacto:** Dijkstra expande siempre el estado de menor costo
acumulado conocido. Cuando un estado se marca como "visitado" (se
expande por primera vez), ya no puede existir un camino más barato hacia
él, esa es la propiedad que garantiza optimalidad, siempre que todos los
pesos de las aristas sean no negativos (que es el caso aquí: son
distancias).

---

### `reg_busqueda.py`

Acumula métricas de cada iteración de la búsqueda (una expansión de
estado a la vez) y las exporta a CSV. No hay "episodios" como en RL —
Dijkstra hace una sola pasada.

| Función | Qué hace |
|---|---|
| `registrar(iteracion, estado, costo_acumulado, tamano_cola)` | Guarda en memoria una fila con el número de iteración, el estado expandido, su costo acumulado, y cuántos elementos quedan en la cola de prioridad. |
| `guardar_csv(ruta)` | Escribe todas las filas acumuladas a un archivo CSV, con encabezados. |

---

### `main.py`

El script que arma todo y lo ejecuta.

| Función | Qué hace |
|---|---|
| `resolver_tsp(problema, registro=None)` | Llama a `dijkstra(problema, registro)` para obtener los costos y padres de todos los estados. Luego busca, entre **todos los estados "meta"** (ciudades ya todas visitadas), cuál da el menor costo total al sumarle el regreso al origen (`costo_cierre_ciclo`). Reconstruye la ruta completa con `reconstruir_camino` y le agrega el regreso final a la ciudad de inicio. |
| Bloque `if __name__ == "__main__"` | Construye el `Grafo` de ciudades arista por arista, crea el `ProblemaTSP`, mide el tiempo de ejecución, corre `resolver_tsp` con un `RegistroBusqueda`, guarda el CSV de la búsqueda, e imprime la ruta óptima, su costo, cuántos estados se expandieron y el tiempo total. |


---

## Resumen del flujo de ejecución

```
main.py
  ├─ Grafo()                         → arma el mapa de ciudades
  ├─ ProblemaTSP(grafo)              → construye la vista de "estados"
  ├─ dijkstra(problema, registro)    → explora el grafo de estados, exacto
  ├─ reconstruir_camino(...)          → arma la secuencia de ciudades
  └─ registro.guardar_csv(...)         → exporta el historial de la búsqueda
```