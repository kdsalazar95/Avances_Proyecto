# Ciclo RL de la CGRA: agente LLM + evaluador

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

# Corrida sin LLM: un sustituto determinista hace el papel de los skills (solo reducción, filas 'M X M' / 'M X' con X en E, V, C)
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

## Estructura

```
rl/
├── entorno/      reglas del problema, compartidas por RL y Dijkstra + Greedy: contexto (A01) y validación de la acción (A03)
├── agente/       lo que propone y aprende: política, recepción, historial, óptimas y ciclo (A02, A04–A08)
├── evaluador/    E01–E08: Skills 01–09 (o el sustituto) + simulador, métricas, recompensa y reporte
├── herramientas/ referencia, informe, recálculo, GIF del log, codex.sh
├── config/       corrida_*.json
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
| A06 | `agente/politica.py` | Política del enunciado π(C, E) = A: `proponer`/`incorporar`; manual, comando (LLM) y guion |
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

- C_A por capacidades declaradas: M = MS+MV = 12, C = CS+CV = 3, E = 1, V = 2, G = 15 (w = 4/8/1/2).
  Referencia 3x3 `M E M / M V M / M E M` (sin PEs C): C_A = 6·12 + 2·1 + 1·2 = 76; con el sustituto, 129 ciclos.
- Regiones de salida comparadas: reducción `c` y `result`; matmul y convolución `result`.
- Si los skills reportan que no pueden mapear, no se reintenta; si generan `.txt` inválidos, se reintenta
  con la lista de errores hasta `N_corr` veces.
- Una configuración repetida (mismo `cgra.json`) se evalúa de nuevo y se marca `duplicada_de`; como los skills
  son un LLM, el resultado puede variar.

## Pendientes

- K, tolerancias, `tolerancia_empate_costo`, `P_fallo` (cota empírica: `resumen.json → avisos_penalizacion`),
  `N_corr`, límites e instancias.
- Probar la corrida principal con `claude` instalado: que los Skills 01–09 compilen la referencia 3x3
  `M E M / M V M / M E M` de cada operación con el prompt de `evaluador/compilador.py`. No se ejecutó aquí.
- Parametrización de los `.c`: `operaciones.c` tiene tamaños fijos (`#define`); una corrida evalúa una instancia.
- El `cgra/src/cgra.json` del árbol de trabajo usa rangos en `filas`/`columnas`: sirve de esquema, pero
  `run_cgra.py` no lo carga directamente (dos pruebas de `cgra/test` fallan por eso). Cada configuración del
  agente trae dimensiones fijas.
