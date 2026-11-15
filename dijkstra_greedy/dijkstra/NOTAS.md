# Bloque Dijkstra: supuestos de hardware

## Evaluador (Dijkstra + Greedy)

`evaluador.py` es la caja "Evaluador" del ciclo de retroalimentación. Frente a la solución RL (`solucion_rl.md`, en la rama de la solución RL) solo cambia quién genera los `PE*.txt`: el bloque Dijkstra en lugar de los Skills 04–07. Todo lo demás sigue el contrato de la sección 10 de ese documento. Las reglas compartidas están en `contrato.py` y los parámetros, con los valores de la corrida RL, en `costos.json`.

Recibe una configuración (un `FormatoCGRA`, la ruta de un `.json` o un diccionario con las claves de `cgra.json`) y sigue estos pasos:

| Paso | Qué hace | Estado si falla |
|---|---|---|
| Validación previa | Las reglas de `rl/agente/paquete.py`: filas 1–32, columnas 3–32, `M` en las columnas oeste y este, interior `E`/`V`, tipos e instrucciones iguales al esquema, memoria ≤ 4096, ancho 4 | `accion_invalida` |
| Mapeo | Dijkstra, por operación | `fallo_evaluador` |
| E03 | `validacion_programas.py`: las reglas de `rl/medicion/validacion_programas.py` (un archivo por PE, registros definidos antes de leerse, cada salida escrita una vez, encabezados) | `fallo_evaluador` |
| Simulación | `cgra/src/run_cgra.py` (el simulador del RL) en otro proceso, con `t_max_simulacion_s` = 60. Ciclos y NOPs de `reporte_ciclos.txt`; memoria final con `--write-back` sobre una copia (`simulacion.py`) | `fallo_simulador` |
| Validación funcional | Cada elemento debe cumplir \|y − ŷ\| ≤ ε_abs + ε_rel·\|y\| | `incorrecta` |

Dijkstra es determinístico: reintentar con los errores (`N_corr`) daría los mismos `.txt`, así que no se reintenta.

| Costo | Cómo se calcula |
|---|---|
| C_A | w₁·n_MS + w₂·n_MV + w₃·n_CS + w₄·n_CV con w = 4, 8, 1, 2: M = 12, E = 1, V = 2, C = 3 |
| C_L | Por operación, `ciclos_x_nops`: ciclos·(1 + NOPs/(ciclos·PEs)) |
| C_R | Por operación, promedio de \|y − ŷ\|/(\|y\| + ε), con valor absoluto |
| C | Promedio sobre las operaciones de K_A·C_A/C_A_ref + K_L·C_L,op/C_L_ref,op + K_R·C_R,op, con K = 1, 1, 1. Solo si todas son `correcta`; si no, C = P_fallo = 10 |

Los normalizadores son los **congelados** de `normalizacion.json`: la referencia `M E M / M V M / M E M` medida una vez con la solución RL (`r05-exploracion`). Una operación que no esté ahí se normaliza con la referencia medida por este evaluador; si la referencia no es correcta, `ErrorReferencia`.

El resultado esperado es el grafo ejecutado en el orden del C sobre la memoria inicial. En las tres operaciones de `operaciones.c` da exactamente los mismos valores y las mismas regiones que `rl/medicion/entradas.py`.

La recompensa r = C_best − C (−P_fallo si se penaliza, con C_best,0 = C_ref) depende de la historia, así que la calcula Greedy al registrar la evaluación.

API interna: `Evaluador().evaluar(configuracion)` devuelve una `Evaluacion` con `estado`, `costo`, `metricas` (C_A, C_L, C_R, C_A_norm, C_L_norm, ciclos, nops, fraccion_nops, capacidades) y `operaciones`. Los grafos se leen una vez y cada configuración ya evaluada se guarda. El ciclo agente-evaluador (`agente_greedy.py`) lo llama con cada configuración que propone Greedy.

Dijkstra sustituye los skills 04–07. Desde Python, `cargar_grafo()` una vez por operación y `evaluar_configuracion(grafo, formato)` por configuración.

| Supuesto del skill | Cómo quedó (según `malla.py`, `programa.py` e `isa.py` del simulador en `cgra/src/`) |
|---|---|
| Tipos MS/MV/CS/CV | Tipos reales de `cgra.json` (M, E, V). Compatibilidad: `compatibilidad.json` traduce la operación a una instrucción y se buscan los tipos que la tienen en `instrucciones`. Malla desde `FormatoCGRA` (`ciclo_filas` o `malla`). |
| Latencia 1 | Se mantiene. |
| El vecino lee el resultado en t+1 | **Cambiado.** Pasar un dato a un vecino es `SEND` + `RECV` en el mismo ciclo y **ocupa los dos PEs**. El dato se usa en el receptor al ciclo siguiente. Por eso, en `M V V E`, el `add` queda en el ciclo 4, no en el 3. |
| Relevo con `MOV` | Relevo con `RECV` y luego `SEND` (el `MOV` del ISA solo carga constantes). |
| Registros ilimitados | **Cambiado.** El texto del ISA solo acepta 5 registros (`rA rB rC rT acc`). El mapeo cuenta los valores vivos por PE y ciclo, y asigna registros por coloreo de intervalos. |
| M reenvía datos | Opción `reenvio_memoria` (`--sin-reenvio-memoria`), activa por defecto. |
| N, S, O, E sin toroide | Se mantiene. |

Cambios del algoritmo respecto al pseudocódigo del skill:

- Dijkstra busca **hacia atrás** desde `(PE destino, ciclo)` con la clave `(-ciclo, saltos, PE)`. Así cada operando sale lo más tarde posible y ocupa registros el menor tiempo.
- `load`, `vload` y `const` se ubican junto con su primer consumidor. Si no, se cargarían cientos de ciclos antes de usarse y llenarían los registros.
- El grafo sale de ejecutar simbólicamente `@<operacion>` en `operaciones.ll`, con los bucles desenrollados. `fmuladd` se separa en MUL + ADD; con un `fneg` (`acc -= x * y`) queda MUL + SUB, y `(float)i` es una constante. Si un `load` lee lo que acaba de escribir el `store` anterior, se usa el valor directamente, y los `store` pisados se eliminan.

## Modo vectorial

`vectorizar.py` agrupa de a `ancho_vector` (4) operaciones iguales: `VLD`/`VST` para posiciones seguidas, `VBCAST` para un escalar común a los carriles (por ejemplo `a[i][k]` en matmul, `kernel[ki][kj]` en convolucion) y `VADD`/`VMUL` carril a carril. Las cadenas de sumas (la suma de reduccion y la columna que sobra en convolucion) se agrupan como `acc + VREDSUM(4 valores)`. Lo que no encaja, como la columna 8 de matmul (cuyos `b[k][8]` no están seguidos), queda escalar.

Los vectores usan los registros `v0`–`v7` (un banco aparte de los escalares) y viajan con `VSEND`/`VRECV`, así que solo pasan por PEs que tengan esas instrucciones: un E no puede reenviar vectores.

`evaluar_configuracion(..., modo=...)` (en la línea de comandos, `--modo`) acepta cuatro valores:

- `auto` (por defecto): vectorial si la CGRA tiene aritmética vectorial; si ese mapeo no es posible, escalar.
- `escalar`.
- `vectorial`.
- `mejor`: prueba los dos y se queda con el de menos ciclos.

`metricas["modo"]` dice cuál se usó.

| 3x3 de `cgra.json` | escalar | vectorial |
|---|---|---|
| reduccion | 404 ciclos | 120 ciclos |
| matmul | 2190 | 543 |
| convolucion | 3891 | 1057 |

## Limitaciones

- El simulador y el evaluador calculan en float de Python (double, float64) por decisión del equipo; el C, en float32. La memoria (`memoria.bin`) sí es float32. Con valores normales solo cambia el redondeo, pero un producto largo que se desborda en el C (`inf`) sigue siendo finito en el simulador, y C_R no lo ve. Las pruebas comparan contra el C real y sí lo detectan.
- Las operaciones de `operaciones.c` pueden ser cualesquiera, siempre que usen datos `float` con `+ - * /`, enteros solo en índices y bucles con límites fijos, sin `if` sobre datos ni llamadas (además de `inicializar_<operacion>`).
- `VREDSUM` cambia el orden de las sumas respecto al C. Con los datos enteros del proyecto el resultado es idéntico; con decimales, el redondeo podría diferir.
- Es heurístico (goloso), no óptimo. El flujo con LLM, con sus tablas hechas a mano, sigue siendo más corto en la 3x3 (76, 441 y 756 ciclos).
- En mallas grandes sigue siendo lento: en modo vectorial, 6x10 u 8x8 tardan de 1,5 a 32 s por operación (convolucion es la más lenta).

## Decisiones de evaluación

Esta es la alternativa Dijkstra + Greedy del PDF. Para que se pueda comparar con la solución RL, comparte con ella las ecuaciones de costo, los estados, las validaciones, la referencia y el formato del registro (`solucion_rl.md`, sección 10). Antes usaba el error relativo con signo y costos sin normalizar; se cambió a lo del RL porque solo deben cambiar el agente y el generador de programas. El presupuesto no es el del RL: Greedy barre todos los tamaños (`config/agente.json`, `max_evaluaciones` y `paciencia` en `null`). Para reproducir el presupuesto de una corrida RL, basta con ponerlos en esos valores.
