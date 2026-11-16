# Dijkstra + Greedy

Es la solución RL con dos piezas cambiadas: el agente es un **Greedy** determinístico y el
evaluador usa **Dijkstra** en lugar de los skills 04-07. Las ecuaciones de costo son las
mismas de la solución RL, para que los resultados se puedan comparar.

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

Un solo comando corre el ciclo completo, desde `dijkstra_greedy/`:

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
   visitó. Con probabilidad `epsilon` **explora**: elige uno de ellos al azar. Si no, **explota**:
   elige el de menor peso (en empate, el primero fila por fila). Repite hasta que el PE actual no
   tiene vecinos sin visitar. No es recursivo y ningún PE se visita dos veces: no hay ciclos.
   Con `epsilon: 0` es Greedy puro; la `semilla` hace que la misma corrida dé el mismo camino.
   El resultado son los PEs del camino; no tienen que formar un rectángulo.
5. **Valuación del camino.** Con los datos de esa misma simulación (sin volver a simular) se aplica
   la función de costo a los PEs del camino: C_A = suma de w_tipo de los PEs del camino; por
   operación, C_L = ciclos·(1 + NOPs del camino/(ciclos·PEs del camino)); C_R el de la simulación;
   C = promedio de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R, con los normalizadores del RL.

## Resultados

| Archivo | Contenido |
|---|---|
| `camino.json` | **El resultado:** los PEs del camino en orden, con su tipo, NOPs, ciclos, peso y la decisión de cada paso (explotación o exploración); cuántos de cada tipo; el peso total; el mapa de la malla con `.` en los PEs que quedan fuera, y la valuación (C, C_A, C_L, C_R por operación) |
| `pesos.json` | El peso de cada PE de la malla y los datos con que se calculó |
| `evaluacion_malla.json` | La evaluación de la malla: estado, C, costos por operación y NOPs por PE del reporte de ciclos |
| `resumen.json` | Malla, su C, PE de inicio, PEs del camino por tipo, peso total, C del camino y tiempo |

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
| `config/agente.json` | Agente: malla del grafo, PE de inicio, `epsilon` y `semilla`, procesos del simulador y del mapeador (`mapeador_procesos`, o `--mapeador-procesos` en la línea de comandos; mismo resultado que el secuencial) y tiempo máximo de simulación |
| `config/compatibilidad.json` | Operación del grafo → instrucción del ISA (para Dijkstra) |

## Estructura

```text
agente_greedy.py      el ciclo agente-evaluador (único comando)
config/               costos, normalizadores, agente y compatibilidad
dijkstra/
  agente.py           agente Greedy: pesos de los PEs y camino por el vecino de menor peso
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
tests/                pruebas (python3 dijkstra_greedy/tests/run_tests.py)
```

`dijkstra/NOTAS.md` explica los supuestos de hardware de Dijkstra y `dijkstra/ARCHIVOS_LL_DOT.md`,
qué toma de los archivos `.ll` y `.dot`.
