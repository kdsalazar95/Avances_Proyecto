# Sistema 1: agente LLM + evaluador con LLM

Es uno de los dos sistemas que se contrastan en el proyecto:

| | Agente | Evaluador (genera los PE*.txt) |
|---|---|---|
| **Sistema 1 (esta carpeta)** | **LLM, sin Greedy** | **Skills 01–09 (LLM)** |
| Sistema 2 (rama `feature/sistema2-greedy`) | ε-greedy, sin LLM | Dijkstra |

Implementa el diseño del proyecto (PDF "Dijkstra + Greedy _ Agente RL" y el
diagrama "Evaluación" del PDF anotado):

- **Agente LLM (política):** propone la **configuración** de la CGRA: malla de
  PEs, tipos de PE y tamaño de red (`cgra.json`).
- **Evaluador, lo que ya estaba hecho:** recibe la configuración y el `.c`. Los
  **Skills 01–09** (LLM, otra sesión) generan los `PE*.txt`; la **parte de
  Python** (`cgra/src/run_cgra.py`) los ejecuta e imprime el reporte por ciclos
  y la memoria. Con eso se calculan C_A, C_L, C_R, el costo y la recompensa,
  que vuelven al agente.

```
 A01 contexto ─▶ A06 política (LLM) ─▶ candidatas/<id>/cgra.json ─▶ A03 validación previa
                                                                     │ inválida: accion_invalida
        ┌──────────── evaluador (rl/evaluador: Skills 01–09 + simulador) ◀──┘ válida
        │ E01 carpeta aislada ─▶ E02 Skills 01–09 ─▶ E03 ¿.txt bien? ─(no, N_corr)─▶ E02 … ─▶ "Reportar"
        │                                              │ sí
        │                     E04 simulador (cgra/src) ─▶ E05 funcional ─▶ E06 métricas ─▶ E07 recompensa
        └──▶ E08 reporte ─▶ A04 recepción ─▶ A05 historial.jsonl ─▶ A07 optimas.json ─▶ A06 …
```

## Uso (desde la raíz del repositorio)

```sh
# Corrida con los Skills 01–09 reales corridos por Codex (rl/herramientas/codex.sh; con Claude: corrida_reduccion_claude.json)
uv run python -m rl iniciar --config rl/config/corrida_reduccion.json

# Solo para pruebas (no es el Sistema 1: no usa los Skills): un sustituto determinista hace el papel de los skills (solo reducción, filas 'M X M' / 'M X')
uv run python -m rl iniciar --config rl/config/corrida_reduccion_sin_llm.json

# El agente: leer el contexto de decisión y escribir candidatas/it-0001/cgra.json (+ candidata.json)
uv run python -m rl prompt  --corrida resultados/corridas/r01-reduccion
uv run python -m rl evaluar --corrida resultados/corridas/r01-reduccion --candidata resultados/corridas/r01-reduccion/candidatas/it-0001

# Ciclo automático: "politica": {"tipo": "comando", "comando": [...]}; o --guion DIR1 DIR2 ... con cgra.json ya escritos
uv run python -m rl ciclo   --corrida resultados/corridas/r01-reduccion
uv run python -m rl resumen --corrida resultados/corridas/r01-reduccion

# Escribir un cgra.json de filas 'M X M' (o 'M X'); por defecto la referencia E, V, E
uv run python -m rl referencia DIR --computo E,V,E [--sin-este]

# Pruebas
uv run python -m unittest discover -s rl/tests -t .
```

## Comparación con el Sistema 2

`rl/config/comparacion.json` deja el Sistema 1 en las mismas condiciones que el Sistema 2:

- **La misma referencia 32x32:** la de `cgra.json` (fila 0 escalar E, fila 1 vectorial V, alternadas, con M
  en las columnas 0 y 31). Es la configuración de partida: su C es `C_best_0` y la primera recompensa se mide
  contra ella.
- **Los mismos normalizadores fijos** (`rl/config/normalizacion.json`, el mismo archivo del Sistema 2). El costo
  es `C = K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R`: dividir entre `C_A_ref` y `C_L_ref` pone el área y la
  latencia en la misma escala. Si cada sistema usara los suyos, un mismo C significaría cosas distintas. La
  referencia se evalúa con ellos (no mide los propios).
- **Los mismos pesos, K, tolerancias, P_fallo y tiempo máximo de simulación (36000 s).**
- **Por episodios, como el Sistema 2** (`agente/episodios.py`): cada episodio empieza en la referencia; cada
  propuesta se evalúa y pasa a ser la mejor del episodio si baja su costo; el episodio termina tras 3 propuestas
  seguidas sin mejorar, si el LLM responde PARAR o a los 40 pasos. La corrida termina con `max_episodios` o con
  `paciencia` episodios seguidos sin bajar el mejor costo. El LLM ve el episodio actual y la mejor configuración
  de cada episodio anterior.
- **Epsilon (exploración, la del Sistema 2):** con probabilidad `epsilon` la acción es un paso al azar desde la
  mejor configuración del episodio: quitar un bloque de 16, 8, 4, 2 o 1 filas o columnas del interior (del inicio o
  del final) o cambiar un PE del interior E <-> V. Si no, decide el LLM. `epsilon = 0` es el LLM puro y
  `epsilon = 1`, pasos al azar. No hay Greedy.
- **Caché, como el Sistema 2:** una configuración ya evaluada no se vuelve a evaluar (la clave es el hardware, no
  el texto del cgra.json). Repetida en la corrida, su evaluación cuesta 0 s; si viene de otra corrida del barrido,
  cuenta lo que tardó cuando se evaluó.

```sh
# Una corrida por epsilon con paciencia 10 y tope de 30 episodios (la referencia 32x32 se evalúa una vez y las
# demás la reusan; las corridas comparten la caché). Paciencia 3 y 6 salen de cortar esas mismas corridas.
uv run python -m rl barrido --config rl/config/comparacion.json --epsilons 0 10 20 80 90 100

uv run python -m rl convergencia --corrida resultados/corridas/s1-comparacion-eps000   # rehace los datos de una corrida
uv run python -m rl comparar DIR1 DIR2 ... --salida DIR                                # junta varias corridas
uv run python -m rl resultados --corrida DIR                                           # tabla, curva y GIF de una corrida
uv run python -m rl graficas COMP_S1 [COMP_S2] --salida DIR                            # gráficas de la comparación
```

Datos que deja (`rl/herramientas/convergencia.py`; el Sistema 2 escribe el mismo formato, con su `sistema`):

| Archivo | Contenido |
|---|---|
| `<corrida>/convergencia.csv` | Una fila por evaluación (n = 0 la referencia): quién la propuso (`llm`, `exploracion`), estado, tamaño, C, C_A, C_A_norm, C_L_norm, C_R, `C_best` hasta ahí, si mejoró, tiempo de decisión, de evaluación y acumulado |
| `<corrida>/convergencia.json` | Resumen: epsilon, motivo de fin, evaluaciones, C_ref, C_best, mejora relativa, **convergencia** (evaluación y tiempo en que se alcanzó el mejor C) e **hitos** del 50, 90 y 100 % de la mejora |
| `<corrida>/episodios.json` | Los episodios: pasos, mejor configuración y su C, si bajó el mejor de la corrida y por qué terminó |
| `<id>-comparacion/comparacion_epsilon.csv` | Una fila por epsilon y paciencia (6 x 3 = 18) con ese resumen: la tabla para comparar tiempos de convergencia |
| `<id>-comparacion/curvas_epsilon.csv` | C y C_best de cada evaluación de cada epsilon y paciencia (formato largo): las curvas de cómo baja la función de costo |
| `<corrida>/configuraciones.csv` (+ `_operaciones.csv`, `.json`) | Tabla de todas las configuraciones distintas, con las columnas de la tabla del Sistema 2: costos, ciclos, NOPs, latencia y error por operación, si es eficiente (Pareto), quién la propuso, en qué episodio y si fue la mejor de su episodio |
| `<corrida>/curva_costos.png`, `exploracion.gif` | El C de cada evaluación en orden, y un cuadro por evaluación con la malla propuesta (colores M, E, V del Sistema 2) y la curva hasta ese punto |
| `<id>-comparacion/*.png` | Las gráficas de la comparación: mejor C por evaluación y por tiempo (una línea por epsilon), tiempo y evaluaciones hasta converger (por epsilon y paciencia) y mejor C por epsilon. Con `python -m rl graficas COMP_S1 COMP_S2 --salida DIR` salen los dos sistemas lado a lado |

**Paciencia sin correr de más.** La paciencia solo decide cuándo se deja de intentar: con la misma semilla, una
corrida con paciencia 3 hace los mismos primeros episodios que una con paciencia 10 (el LLM solo ve lo que ya
pasó) y se detiene antes. `convergencia.corte` corta cada corrida donde se habrían detenido las de paciencia 3 y 6
(una prueba compara el corte con una corrida real de menor paciencia).

El tiempo acumulado cuenta la decisión (el LLM o el azar) y la evaluación (Skills + simulador) desde la
primera propuesta; la referencia es n = 0, tiempo 0.

## Estructura

```
rl/
├── entorno/      reglas del problema: contexto (A01) y validación de la acción (A03)
├── agente/       lo que propone y aprende: política, recepción, historial, óptimas, ciclo y episodios (A02, A04–A08)
├── evaluador/    E01–E08: Skills 01–09 (o el sustituto) + simulador, métricas, recompensa y reporte
├── herramientas/ convergencia y barrido de epsilon (comparación), referencia, informe, recálculo, GIF del log, codex.sh
├── config/       comparacion.json + normalizacion.json (la comparación) y corrida_*.json
└── tests/
```

Las corridas se escriben en `resultados/corridas/<id>/` (no se versionan). El diseño completo está en
`docs/solucion_rl.md` y los formatos compartidos en `docs/formatos.md`.

## Skill → módulo

| Skill | Módulo (`rl/…`) | Qué hace |
|---|---|---|
| A01 | `entorno/contexto.py` | Contexto: tipos/instrucciones del `cgra.json` real, límites de la corrida, regiones de memoria |
| A02 | LLM (prompt de `agente/politica.py`) | Escribe `cgra.json` + `candidata.json` |
| A03 | `entorno/validacion_accion.py` | Validación previa: `FormatoCGRA`, rango, tipos reales, memoria en bordes, memoria y cómputo |
| A04 | `agente/recepcion.py` | Verifica id, hash del `cgra.json` y coherencia del reporte |
| A05 | `agente/memoria.py` | Historial (solo agregar), ya evaluada, reconstrucción, mejor candidato y entrada de la política π(C, E) |
| A06 | `agente/politica.py` | Política del enunciado π(C, E) = A: `proponer`/`incorporar`; manual, comando (LLM) y guion; `PoliticaEpsilon` (con probabilidad epsilon, una configuración al azar) |
| A07 | `agente/optimas.py` | Óptimas: mejora estricta, empates, duplicados |
| A08 | `agente/ciclo.py` | Referencia, ciclo, paciencia, reanudación, `resumen.json` |
| E01 | `evaluador/preparacion.py` | Carpeta `eval/<id>/` con el `cgra.json` y la memoria inicial |
| E02 | `evaluador/compilador.py`, `evaluador/compilador_sustituto.py` | **Skills 01–09**: `CompiladorSkills` (LLM por comando) o `CompiladorReduccion` (sustituto) |
| E03 | `evaluador/validacion_programas.py`, `evaluador/evaluacion.py` (`compilar_con_skills`) | ¿Están bien los `.txt`? Cargador real + chequeos; reintentos `N_corr` o "Reportar" |
| E04 | `evaluador/simulacion.py` | `run_cgra.py` en subproceso con timeout; ciclos y NOP del reporte |
| E05 | `evaluador/funcional.py`, `evaluador/entradas.py` | Esperado en CPU y comparación elemento a elemento |
| E06 | `evaluador/metricas.py` | C_A del `cgra.json`, C_L = ciclos, C_R, costo normalizado |
| E07 | `evaluador/recompensa.py` | `r = C_best_prev − C`, `−P_fallo`, o `None` |
| E08 | `evaluador/reporte.py` | Reporte + `comprobar` |

## Estados

| Estado | Cuándo | Costo / recompensa |
|---|---|---|
| `accion_invalida` | el `cgra.json` no pasa A03 | `P_fallo` / `−P_fallo` |
| `fallo_evaluador` | los Skills 01–09 reportan, o los `.txt` siguen mal tras `N_corr` correcciones | `P_fallo` / `−P_fallo` |
| `fallo_simulador` | timeout, bloqueo, instrucción inválida, excepción | `P_fallo` / `−P_fallo` |
| `incorrecta` | el resultado no pasa la validación funcional | `P_fallo` / `−P_fallo` (C_R medido se conserva) |
| `correcta` | todo bien | costo medido / `C_best_prev − C` |
| `error_infraestructura` | no hay LLM, se cayó el entorno | sin costo ni recompensa; no consume paciencia |

## Qué se reutiliza y qué es nuevo

- **Reutilizado sin cambios:** Skills 01–09 (`cgra/agent_skills/`), `FormatoCGRA`, `CGRA`, `Cargador`,
  `Programa`, `ImagenMemoria`, `run_cgra.py` y `operaciones.c` (genera `memoria.bin`).
- **Nuevo:** todo `rl/`. `rl/evaluador/` no reimplementa los Skills ni el simulador: los llama y calcula lo que
  el PDF pide (correctitud, C_A, C_L, C_R, costo, recompensa). `evaluador/compilador_sustituto.py` es el
  **sustituto** de los skills para pruebas y `herramientas/referencia.py` escribe `cgra.json` de filas
  `M X M`/`M X`; la corrida principal usa los skills reales.

## Decisiones tomadas (configurables en `corrida.json`)

- C_A por capacidades declaradas: M = MS+MV = 12, E = 1, V = 2 (w = 4/8/1/2). Sin PEs C ni G.
  Referencia 32x32 con filas alternas V/E (M en columnas 0 y 31; el interior sigue el tipo de fila).
- Regiones de salida comparadas: reducción `c` y `result`; matmul y convolución `result`.
- Si los skills reportan que no pueden mapear, no se reintenta; si generan `.txt` inválidos, se reintenta
  con la lista de errores hasta `N_corr` veces.
- Una configuración repetida (mismo `cgra.json`) se evalúa de nuevo y se marca `duplicada_de`; como los skills
  son un LLM, el resultado puede variar.

## Pendientes

- K, tolerancias, `tolerancia_empate_costo`, `P_fallo` (cota empírica: `resumen.json → avisos_penalizacion`),
  `N_corr`, límites e instancias.
- Probar la corrida principal con `claude` instalado: que los Skills 01–09 compilen la referencia 32x32
  (filas alternas V/E) de cada operación con el prompt de `evaluador/compilador.py`. No se ejecutó aquí.
- Parametrización de los `.c`: `operaciones.c` tiene tamaños fijos (`#define`); una corrida evalúa una instancia.
- El `cgra/src/cgra.json` del árbol de trabajo usa rangos en `filas`/`columnas`: sirve de esquema, pero
  `run_cgra.py` no lo carga directamente (dos pruebas de `cgra/test` fallan por eso). Cada configuración del
  agente trae dimensiones fijas.
