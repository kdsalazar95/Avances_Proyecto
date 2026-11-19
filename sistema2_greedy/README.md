# Sistema 2: agente Greedy + evaluador Dijkstra

Es uno de los dos sistemas que se contrastan en el proyecto:

| | Agente | Evaluador (genera los PE*.txt) |
|---|---|---|
| Sistema 1 (`rl/`) | LLM, sin Greedy | Skills 01-09 (LLM) |
| **Sistema 2 (esta carpeta)** | **ε-greedy, sin LLM** | **Dijkstra** (sustituye los skills 04-07) |

Este sistema **no usa ningún LLM**: Greedy explota y, con probabilidad `epsilon`, se explora con
una acción al azar. Las ecuaciones de costo, el simulador y los normalizadores son los mismos del
Sistema 1, para que los resultados se puedan comparar.

## Ciclo de retroalimentación

```text
                      reporte de ciclos de cada PE
   ┌──────────────────────────────────────────────────┐
   │                                                  │
   ▼                                                  │
Agente Greedy ── malla (32x32) ──▶ Evaluador ◀── operaciones.c
  pesos de los PEs                     ├─ skills 01-03  entradas, IR + memoria.bin, grafos .dot
  camino por los PEs                   ├─ Dijkstra      sustituye los skills 04-07
                                       ├─ skills 08-09  PE*.txt y su validación (E03)
                                       ├─ simulador     cgra/src/run_cgra.py (reporte_ciclos.txt)
                                       └─ costos        C_A, C_L, C_R → C
```

Un solo comando corre el ciclo completo, desde `sistema2_greedy/`:

```bash
python3 agente_greedy.py                          # resultados en resultados/greedy/
python3 agente_greedy.py --salida resultados/otra  # otra carpeta (no pisa resultados anteriores)
python3 agente_greedy.py --continuar --salida resultados/greedy   # reusa la simulación guardada
```

1. **Skills 01-03.** Compila `cgra/operaciones/operaciones.c`: `operaciones.ll` y `memoria.bin`
   con `clang-18`, y los grafos `.dot` con `opt-18`, en `cgra/programas/<operacion>/`. Solo
   recompila si falta algo o si cambió el C (`--forzar-compilacion` para repetirlo).
2. **Una simulación de la malla.** La malla del grafo (`malla` en `config/agente.json`; por
   defecto la más pesada, 32x32, con la distribución de `cgra.json`) se mapea con Dijkstra y se
   simula una vez, cada operación en su propio proceso. En 32x32 tarda horas; queda guardada en
   `evaluacion_malla.json` y `--continuar` la reusa.
3. **Pesos de los PEs.** El grafo de Greedy es la malla: cada **vértice es un PE** y cada
   **arista es la conexión con un vecino** norte, sur, este u oeste. El peso de un PE es
   `w_tipo · (1 + NOPs/ciclos)`: `w_tipo` es el área de su tipo con los pesos w1-w4 de
   `config/costos.json` (M = 12, E = 1, V = 2), y NOPs y ciclos salen del `reporte_ciclos.txt`
   del simulador, sumados sobre las operaciones. Un PE que pasa más tiempo en NOP pesa más.
4. **Camino epsilon-greedy.** Empieza en PE00 (`inicio`) y revisa los vecinos que todavía no
   visitó. Con probabilidad `epsilon` **explora**: uno de ellos al azar. Si no, **explota** (greedy): elige el vecino que deja el **menor costo C** del camino, con
   la función de costo del paso 5. Un camino que no cumple los tipos de PE (memoria y cómputo, E y
   V) cuesta `P_fallo`, como en el contrato; en empate, el de menor C valuado y luego el primero fila
   por fila. **Paro:** termina cuando el camino ya cumple los tipos de PE y ningún vecino libre baja
   su C ("si el costo sube, se descarta"), o cuando el PE actual no tiene vecinos sin visitar
   (`parada`: `sin_mejora` o `sin_vecinos`). **Seguir o parar:** en el punto de paro, con probabilidad
   `epsilon`, el azar elige seguir de 0 a `max_extension` (10) PEs más. El estimado usa la latencia de la
   malla completa y siempre prefiere menos PEs; así se pueden probar configuraciones más grandes. Durante esos n pasos no se revisa el paro; queda en `extensiones`.
   En cada paso se guarda la recompensa
   `r_t = C(t−1) − C(t)`. No es recursivo y ningún PE se visita dos veces: no hay ciclos.
   El peso de los PEs (paso 3) ya no decide el camino: queda en `pesos.json`, como dato.
   Con `epsilon: 0` es Greedy puro; la `semilla` hace que la misma corrida dé el mismo camino.
   El resultado son los PEs del camino; no tienen que formar un rectángulo.
5. **Valuación del camino.** Con los datos de esa misma simulación (sin volver a simular) se aplica
   la función de costo a los PEs del camino: C_A = suma de w_tipo de los PEs del camino; por
   operación, C_L = ciclos·(1 + NOPs del camino/(ciclos·PEs del camino)); C_R el de la simulación;
   C = promedio de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R, con los normalizadores del RL.
   Es un **estimado**: los ciclos y NOPs son los de la malla completa.
6. **Episodio (simulación del camino).** El camino terminado se convierte en una malla simulable:
   el **rectángulo** de filas y columnas que ocupa, sin las columnas que ahí son solo de memoria, con
   una columna M al oeste y otra al este (regla de bordes de `costos.json`). Los PEs del rectángulo
   que no están en el camino conservan su tipo. Por ejemplo, PE0000 (M), PE0001 (E), PE0101 (V) da
   `M E M / M V M`. Ese rectángulo se simula **una vez por episodio** con el mismo evaluador: su C es el
   costo **real** del episodio, y la recompensa es `R = C_best_prev − C` (`−P_fallo` si no es
   correcta), como en la solución RL, con `C_best_prev` = el mejor C real de los episodios anteriores
   (al principio, el C de la malla completa).
**Modo de exploración** (`modo` en `config/agente.json`). Por defecto es `bloques`: en lugar de los
pasos 4 a 6, cada episodio es un **descenso desde la malla de referencia**. Se quitan bloques de filas o de
columnas interiores (del inicio o del final; tamaños `bloques` = 16, 8, 4, 2, 1), manteniendo las columnas
M en los bordes y al menos una fila E y una V. Greedy elige la acción de menor C estimado; con probabilidad
`epsilon` se elige una acción al azar. Cada
malla propuesta **se simula**: si su C real baja respecto a la actual se acepta y el descenso sigue con el
mismo bloque; si sube se descarta y se prueba otra acción; sin acciones que mejoren se pasa al bloque
siguiente, y el descenso termina cuando ni el bloque de 1 mejora. Las acciones que darían la misma malla
se prueban una sola vez. Con `tipos: true` sigue la **fase de tipos**: sobre la malla que quedó, cada
acción cambia un PE del interior de E a V o de V a E (los M de los bordes no cambian). Greedy elige el cambio de menor C estimado (el
cambio de área es exacto; la latencia se supone igual) y con probabilidad `epsilon` se elige un cambio al azar; cada malla se simula y se acepta si su C real baja, hasta que ningún cambio mejora o hasta
`max_cambios_tipo` simulaciones. Todas las mallas simuladas quedan en la caché (`evaluaciones_episodios.json`, se
guarda tras cada simulación) y son los puntos del frente de Pareto de la exploración
(`python3 herramientas/pareto.py resultados/<corrida> --exploracion`). Las mallas pequeñas se nombran con la
posición de cada tipo de PE: el interior fila por fila entre las columnas M (`2x4 [VE/VV]` = `M V E M / M V V M`).
`python3 herramientas/regenerar_informes.py resultados/<corrida>` rehace tablas, gráficas, GIF y reporte de
una corrida terminada sin simular. `modo: "camino"` es el modo de los
pasos 4 a 6. Al terminar, la corrida escribe la **tabla completa de resultados** de todas las configuraciones
evaluadas (`configuraciones.csv`, una fila por configuración; `configuraciones_operaciones.csv`, una fila por
configuración y operación con las instrucciones por tipo; `configuraciones.json`): costos (C, C_A, C_L, C_R),
y por operación ciclos (total, cómputo, comunicación, inactivos), instrucciones (memoria, cálculo,
comunicación, otras y por instrucción), NOPs, lecturas, escrituras, transferencias, latencia, error y
correctitud, si es eficiente (Pareto) y en qué orden y por quién se propuso. Y las gráficas: `frente_pareto.png`
(soluciones eficientes y dominadas con la frontera), `curva_costos.png` (C real de cada malla probada),
`curva_episodios.png` y `exploracion.gif` (la malla de PEs durante toda la reducción por bloques). Cada simulación se informa en la terminal al empezar, con un aviso cada 60 s mientras corre
(`--latido`), y al terminar.

7. **Ciclo de episodios (retroalimentación).** Los pasos 4 a 6 se repiten: cada episodio `n` usa la
   semilla `semilla + n`. Si el C real de un episodio baja el mejor (`C_best`), ese episodio pasa a ser
   el mejor; si no, se descarta (pág. 5 del PDF). El ciclo termina con `max_episodios` (10) o cuando pasan
   `paciencia` (3) episodios seguidos sin mejora, como en `rl/`; un fallo de infraestructura no cuenta.
   El C real de cada episodio es la retroalimentación del evaluador: decide cuál es el mejor y cuándo
   se acaba la paciencia. Un rectángulo ya
   simulado no se vuelve a simular (`repetido` si ya salió en la corrida); las simulaciones quedan en
   `evaluaciones_episodios.json` y `--continuar` las reusa. `camino.json`, `historia.json` y
   `resumen.json` son los del mejor episodio (o del último, si ninguno fue correcto).

## Resultados

| Archivo | Contenido |
|---|---|
| `camino.json` | **El resultado:** los PEs del camino en orden, con su tipo, NOPs, ciclos, peso y la decisión de cada paso (explotación o exploración); cuántos de cada tipo; el peso total; el mapa de la malla con `.` en los PEs que quedan fuera (del mejor episodio), la valuación (C, C_A, C_L, C_R por operación), `parada` y `episodio` (malla del rectángulo, estado, C real, C estimado, `C_best_prev` y recompensa) |
| `pesos.json` | El peso de cada PE de la malla y los datos con que se calculó |
| `evaluacion_malla.json` | La evaluación de la malla: estado, C, costos por operación y NOPs por PE del reporte de ciclos |
| `episodios.json` | El ciclo de episodios: por episodio, semilla, camino, decisiones, rectángulo, estado, C real, C estimado, `C_best_prev`, recompensa, si mejoró (`mejora`) y si se repitió; el mejor, `C_best` y por qué terminó (`motivo_fin`) |
| `evaluaciones_episodios.json` | Las simulaciones de los rectángulos (configuración y evaluación); `--continuar` las reusa |
| `historia.json` | El camino paso a paso, para graficarlo después: en cada paso, el PE agregado (fila, columna, tipo, peso), quién lo eligió (`explotacion` = greedy por menor C, `exploracion` = al azar) y el costo del prefijo hasta ese paso (C, C_A, C_L_norm, C_R y el costo de cada operación), sin volver a simular; `C_efectivo` (C, o `P_fallo` si no cumple los tipos) y `recompensa` = C_efectivo anterior − C_efectivo; `parada`, por qué terminó; si el prefijo cumple los tipos de PE (memoria y cómputo, E y V), y `mejor`, el prefijo de menor C entre los que los cumplen (informativo: el resultado sigue siendo el camino completo). Trae la malla, para dibujarla sin otros archivos |
| `resumen.json` | Malla, su C, PE de inicio, PEs del camino por tipo, peso total, C del camino (estimado), parada, episodios, `motivo_fin`, episodio mejor y `C_best`, malla y estado del episodio elegido, su C real y su recompensa, y tiempo |

### Curva de costo y GIF

Después de la corrida, sin simular (desde `sistema2_greedy/`):

```bash
python3 herramientas/gif_camino.py resultados/<corrida>   # curva_costo.png y camino.gif en esa carpeta
```

- `curva_costo.png`: C del prefijo del camino en cada paso, con los pasos de exploración marcados y el de menor C válida.
- `camino.gif`: un cuadro por paso, con la malla (los PEs del camino resaltados) y la curva hasta ese paso.
- `curva_episodios.png` (si hay `episodios.json`): por episodio, el C real (simulado) y el C estimado, el C_best y el
  desglose del C real en área, latencia y error, con el tamaño de cada rectángulo.
- `episodios.gif` (si hay `episodios.json`): un cuadro por episodio, con el camino y su rectángulo sobre la esquina de
  la malla y la curva de C real hasta ese episodio. `--sin-episodios` omite estos dos.

Es determinística: la misma `historia.json` da los mismos archivos, byte por byte. Opciones: `--ms` (por
cuadro), `--max-cuadros` (con caminos más largos salta pasos, siempre incluye el último), `--sin-gif`,
`--salida`. Si la corrida es anterior a `historia.json`, la reconstruye desde `camino.json`,
`evaluacion_malla.json` y `resumen.json` y la guarda.

## Costo (igual que la solución RL)

Lo calcula `dijkstra/contrato.py` con los parámetros de `config/costos.json`.

- **C_A** = w1·n_MS + w2·n_MV + w3·n_CS + w4·n_CV, con w3 < w4 < w1 < w2 = 1, 2, 4, 8.
  Por tipo de PE: M = 12, E = 1, V = 2.
- **C_L**, por operación = ciclos·(1 + NOPs/(ciclos·PEs)): más ciclos o más NOPs cuestan más.
- **C_R**, por operación = promedio de |esperado − obtenido|/(|esperado| + ε).
- **C** = promedio sobre las operaciones de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R, con
  K = 1, 1, 1. Solo cuenta si todas las operaciones son `correcta`; si no, C = P_fallo = 10.

Los normalizadores C_A_ref y C_L_ref son los de la referencia `M E M / M V M / M E M` medida con
la solución RL (`config/normalizacion.json`). Esa referencia es solo para normalizar; el punto de
partida de la búsqueda es la 32x32.

## Configuración

| Archivo | Qué define |
|---|---|
| `config/costos.json` | Ecuaciones: pesos, K, tolerancias, P_fallo, restricciones del espacio de acciones, referencia de normalización |
| `config/normalizacion.json` | C_A_ref y C_L_ref por operación (medidos con la solución RL) |
| `config/agente.json` | Agente: `modo` (`bloques` o `camino`) y `bloques`, malla del grafo, PE de inicio, `epsilon` (0.3) y `semilla`, ciclo de episodios (`max_episodios`, `paciencia`), procesos del simulador y del mapeador (`mapeador_procesos`, o `--mapeador-procesos` en la línea de comandos; mismo resultado que el secuencial) y tiempo máximo de simulación |
| `config/compatibilidad.json` | Operación del grafo → instrucción del ISA (para Dijkstra) |

## Estructura

```text
agente_greedy.py      el ciclo agente-evaluador (único comando)
config/               costos, normalizadores, agente y compatibilidad
dijkstra/
  agente.py           agente Greedy: pesos de los PEs y camino por el vecino de menor costo C
  evaluador.py        evaluador: Dijkstra, PE*.txt, E03, simulador y costos
  contrato.py         ecuaciones y validaciones compartidas con la solución RL
  compilar.py         skills 02-03 (clang-18, opt-18)
  ir_llvm.py          skill 04: grafo de datos desenrollado desde operaciones.ll
  vectorizar.py       skills 04-05: agrupación vectorial (VLD, VADD, VREDSUM...)
  hardware.py         skills 05-06: malla y compatibilidad por tipo de PE
  mapeador.py         skill 07: Dijkstra en la malla extendida en el tiempo
  verificador.py      revisión independiente del mapeo
  salida.py           skill 08: PE*.txt
  validacion_programas.py   skill 09 (E03): validación de los PE*.txt
  simulacion.py       el simulador de cgra/src/ en otro proceso
  dfg.py              grafo de datos y resultado esperado
tests/                pruebas (python3 sistema2_greedy/tests/run_tests.py)
```

`dijkstra/NOTAS.md` explica los supuestos de hardware de Dijkstra y `dijkstra/ARCHIVOS_LL_DOT.md`,
qué toma de los archivos `.ll` y `.dot`.
