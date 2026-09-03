# Solución del TSP con Reinforcement Learning (Q-learning)

## Idea general

A diferencia de Dijkstra (que explora el grafo de estados de forma
exhaustiva y exacta), el enfoque de **Reinforcement Learning** aprende
por prueba y error: un **agente** interactúa con un **entorno**,
moviéndose de ciudad en ciudad, recibiendo una recompensa (negativa,
igual a la distancia recorrida) en cada paso, y ajustando su
conocimiento poco a poco a lo largo de muchos episodios de práctica.

Se usa **Q-learning tabular**, el algoritmo de RL más simple para
espacios de estados y acciones discretos, no hay redes neuronales, se
aprende directamente una tabla `Q[estado, accion]` que estima "qué tan
buena es tomar esta acción desde este estado". Con suficiente
entrenamiento, esta tabla converge hacia el óptimo pero, a diferencia
de Dijkstra, **no hay garantía matemática** de que siempre lo alcance:
depende de los hiperparámetros y de cuánto se entrene.

El estado sigue siendo el mismo que en la rama de Dijkstra:
`(ciudad_actual, mascara_de_ciudades_visitadas)`, la diferencia no está
en qué es un estado, sino en **cómo se explora** el espacio de estados
(exhaustivo vs. aprendido).

## Estructura de archivos

```
comun/grafo.py            → representación pura del mapa de ciudades (compartida con Dijkstra/)
RL/entorno_tsp.py         → interfaz tipo "entorno" (reset/paso/acciones_validas)
RL/agente.py              → el agente: decide (política) y aprende (regla de Q-learning)
RL/ql.py                  → bucle de entrenamiento genérico (agente <-> entorno)
RL/registro.py            → registra métricas de cada episodio y las exporta a CSV
RL/config_agente.json     → todos los hiperparámetros ajustables, sin tocar código
RL/train.py               → arma todo, entrena y guarda resultados
RL/plot.py                → lee el CSV del entrenamiento y grafica el progreso
```

La idea de diseño es la misma que en la rama de Dijkstra: separar el
**entorno** (representa el problema) del **agente** (decide y aprende) y
del **bucle de entrenamiento** (los hace interactuar). Esto significa que
`ql.py` no sabe nada de ciudades, y `agente.py` no sabe nada de TSP,
solo trabajan con estados y acciones genéricos.

---

## Descripción de scripts

### `entorno_tsp.py`

Envuelve el mismo `Grafo` de ciudades que usa Dijkstra, pero con una
interfaz distinta, la que necesita un agente para interactuar paso a
paso, en vez de la exploración exhaustiva que usa Dijkstra.

| Función | Qué hace |
|---|---|
| `reset()` | Empieza un episodio nuevo: vuelve el estado interno a `(ciudad_inicio, mascara_con_solo_esa_ciudad)` y lo devuelve. |
| `acciones_validas(estado)` | Ciudades a las que el agente se puede mover desde ese estado. Si ya se visitaron todas, la única acción válida es volver a la ciudad de inicio (para cerrar el ciclo). |
| `paso(accion)` | Ejecuta el movimiento hacia `accion` (una ciudad). Calcula la recompensa como `-distancia` (negativa, porque el agente maximiza recompensa y nosotros queremos minimizar distancia). Si la acción cierra el ciclo (todas visitadas + regresa al inicio), marca `terminado=True`. Devuelve `(nuevo_estado, recompensa, terminado)`. |
| `nombres_de_ruta(ruta_indices)` | Traduce índices de ciudades a nombres legibles. |

---

### `agente.py` — clase `AgenteQLearning`

Encapsula toda la lógica de decisión y aprendizaje. No sabe nada de
TSP: solo recibe estados y acciones genéricos.

| Método | Qué hace |
|---|---|
| `__init__(alpha, gamma, epsilon_inicial, epsilon_minimo, decaimiento_epsilon, semilla)` | Guarda los hiperparámetros y crea la tabla `Q` vacía (un `defaultdict(float)`: cualquier par `(estado, accion)` no visto todavía vale 0.0 por defecto). |
| `elegir_accion(estado, acciones_validas)` | Política **epsilon-greedy**: con probabilidad `epsilon` elige una acción al azar (**explorar**); si no, elige la de mayor valor `Q` conocido (**explotar** lo aprendido). |
| `mejor_accion(estado, acciones_validas)` | Devuelve la acción de mayor valor `Q` para ese estado, sin ninguna exploración — se usa tanto dentro de `elegir_accion` como al final, para ejecutar la política ya aprendida. |
| `actualizar(estado, accion, recompensa, nuevo_estado, acciones_siguientes, terminado)` | Aplica la fórmula clásica de Q-learning: `Q(s,a) += alpha * (recompensa + gamma * max Q(s',a') - Q(s,a))`. Si el episodio terminó, no hay estado futuro que considerar, así que el término `max Q(s',a')` se toma como 0. |
| `decaer_epsilon()` | Reduce `epsilon` multiplicándolo por `decaimiento_epsilon`, sin bajar nunca del `epsilon_minimo` (para que el agente nunca deje de explorar del todo). |
| `desde_config(config)` (classmethod) | Crea un agente a partir de un diccionario de hiperparámetros (típicamente cargado desde `config_agente.json`). |

**La fórmula de actualización, en palabras:** el nuevo valor de `Q(s,a)`
se mueve una fracción (`alpha`) de la distancia que hay entre su valor
actual y el "objetivo", lo que realmente se obtuvo (`recompensa`) más lo
mejor que se puede esperar del estado al que se llegó
(`gamma * max Q(s',a')`).

---

### `ql.py` — el bucle de entrenamiento

Es el módulo más simple: no toma ninguna decisión, solo hace interactuar
al agente con el entorno. No sabe nada de TSP ni de la fórmula de
Q-learning, eso vive en `agente.py`.

| Función | Qué hace |
|---|---|
| `entrenar(entorno, agente, episodios, registro=None)` | Corre `episodios` episodios completos. En cada uno: `reset()` el entorno, y mientras no termine, pide al agente una acción (`elegir_accion`), la ejecuta en el entorno (`paso`), y le pasa el resultado al agente para que aprenda (`actualizar`). Al final de cada episodio, si se pasó un `registro`, guarda sus métricas (recompensa total, costo, pasos, epsilon usado), y llama a `agente.decaer_epsilon()`. |
| `extraer_politica_greedy(agente, entorno)` | Recorre el entorno usando siempre `agente.mejor_accion(...)` (sin exploración, sin aprender) para obtener la ruta final que el agente considera óptima, y su costo total. |

---

### `registro.py` — clase `RegistroEntrenamiento`

Acumula las métricas de cada episodio y las exporta a CSV, para poder
graficar el progreso del entrenamiento después.

| Método | Qué hace |
|---|---|
| `registrar(episodio, recompensa_total, costo_total, pasos, epsilon)` | Guarda en memoria una fila con esos cinco valores. |
| `guardar_csv(ruta)` | Escribe todas las filas acumuladas a un archivo CSV. |
| `resumen(cada_n)` | Devuelve solo las filas cuyo número de episodio es múltiplo de `cada_n`, útil para imprimir un resumen corto en consola sin mostrar miles de líneas. |

---

### `train.py`

El script de entrenamiento.

| Paso | Qué hace |
|---|---|
| 1 | Carga los hiperparámetros desde `config_agente.json`. |
| 2 | Construye el `Grafo` de ciudades (idéntico al que usa Dijkstra, para poder comparar resultados). |
| 3 | Crea el `EntornoTSP` sobre ese grafo. |
| 4 | Crea el `AgenteQLearning` a partir de la sección `"agente"` del JSON. |
| 5 | Llama a `entrenar(entorno, agente, episodios, registro)`. |
| 6 | Guarda `historial_entrenamiento.csv` con el progreso completo. |
| 7 | Llama a `extraer_politica_greedy` para obtener la ruta final aprendida, la imprime junto con su costo, y muestra un resumen del progreso cada 500 episodios. |

Todas las rutas de archivo (`config_agente.json`, el CSV de salida) se
construyen relativas a la ubicación del propio script, no al directorio
desde donde se ejecuta, así `train.py` funciona sin importar desde qué
carpeta se corra.

---

### `plot.py`

Script de visualización, independiente de `train.py`: solo lee el CSV
ya generado, nunca vuelve a entrenar nada.

| Función | Qué hace |
|---|---|
| `leer_historial(ruta_csv)` | Lee el CSV y separa sus columnas (`episodio`, `costo_total`, `epsilon`) en listas. |
| `promedio_movil(valores, ventana)` | Suaviza una serie de valores promediando cada punto con los `ventana` anteriores, para que la tendencia se vea clara entre el ruido episodio a episodio. |
| `graficar(ruta_csv, ruta_salida)` | Dibuja dos paneles con `matplotlib`: arriba, el costo por episodio (crudo y suavizado); abajo, cómo decae `epsilon` a lo largo del entrenamiento. Guarda el resultado como imagen PNG. |

---

## Resumen del flujo de ejecución

```
train.py
  ├─ config_agente.json           → hiperparametros
  ├─ Grafo()                       → arma el mapa de ciudades (mismo que Dijkstra)
  ├─ EntornoTSP(grafo)               → interfaz reset/paso/acciones_validas
  ├─ AgenteQLearning.desde_config()   → crea el agente
  ├─ entrenar(entorno, agente, ...)     → episodios de interaccion + aprendizaje
  ├─ registro.guardar_csv(...)            → historial_entrenamiento.csv
  └─ extraer_politica_greedy(...)           → ruta final aprendida

plot.py
  └─ lee historial_entrenamiento.csv → progreso_entrenamiento.png
```
