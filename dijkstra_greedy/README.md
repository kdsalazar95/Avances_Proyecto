# Dijkstra + Greedy por etapas independientes

Cada comando ejecuta solo su etapa. No hay un comando que compile, mapee,
simule y evalúe automáticamente. Ejecute los comandos desde `dijkstra_greedy/`.

Estructura del repositorio:

```text
cgra/               compartido con la solución RL
  src/              simulador (run_cgra.py, cgra.json, ...), sin modificar
  operaciones/      operaciones.c y memoria_cgra.h
  programas/        lo genera la etapa 1 (IR y memoria.bin)
  agent_skills/     skills 01-09 de la solución RL
  test/             pruebas del simulador (python3 cgra/test/run_tests.py)
dijkstra_greedy/    esta solución
  *.py              una etapa por script
  dijkstra/         paquete: Dijkstra, evaluador, contrato y agente Greedy
  config/           costos.json, agente.json, normalizacion.json, compatibilidad.json
  tests/            pruebas (python3 dijkstra_greedy/tests/run_tests.py)
```

## 1. Compilación

```bash
python3 compilar_cgra.py todas
```

Genera `cgra/programas/<operacion>/operaciones.ll` y `memoria.bin`. Use `--forzar`
para reconstruir. Debe repetir esta etapa cuando cambie el C o sus cabeceras.

## 2. Propuesta inicial de Greedy

```bash
python3 agente_greedy.py --salida resultados/greedy
```

Solo propone hardware. La primera propuesta es la referencia `M E M / M V M / M E M`
(`config/costos.json → referencia`), como en la solución RL: C_best,0 = C_ref. Escribe
`propuestas/config_*.json` y conserva el estado en `estado.json`. No lanza procesos
de evaluación.

## 3. Mapeo con Dijkstra

Ejemplo para una propuesta, con las tres operaciones:

```bash
python3 mapear_dijkstra.py todas --cgra resultados/greedy/propuestas/config_0001.json --salida resultados/prueba_001
```

Lee el IR y la memoria existentes y mapea. `--salida` es la carpeta de evaluación; con
`todas` hay una por operación en `<salida>/<operacion>/`, con la estructura del RL:

```text
resultados/prueba_001/reduccion/
  cgra.json            la configuración, con tamaño fijo y malla explícita
  memoria.bin          copia de cgra/programas/reduccion/memoria.bin
  pe_instructions/     PE*.txt de Dijkstra, validados con las reglas E03 del RL
```

No ejecuta el simulador. Si Dijkstra no puede mapear o los `.txt` no pasan E03, termina con
código 1 y la carpeta solo tiene `fallo.json`. Esa operación es `fallo_evaluador`: no la
simule y entregue la carpeta igual al evaluador.

## 4. Simulación

El simulador es el de `cgra/src/`, el mismo de la solución RL. Desde `dijkstra_greedy/`:

```bash
timeout 60 python3 ../cgra/src/run_cgra.py resultados/prueba_001/reduccion --cgra resultados/prueba_001/reduccion/cgra.json --memoria resultados/prueba_001/reduccion/memoria.bin --write-back
```

Usa `pe_instructions/` de la carpeta y escribe ahí `reporte_ciclos.txt`. Con `--write-back`,
la memoria final queda en el `memoria.bin` de la carpeta, que es una copia y no la de
`cgra/programas/`. Repita para `matmul` y `convolucion`.

El tiempo máximo del contrato es `t_max_simulacion_s` = 60 s (`config/costos.json`). Si el
simulador lo supera o termina con error, esa operación es `fallo_simulador`: pásela al
evaluador con `--fallo OPERACION=fallo_simulador[:detalle]`.

## 5. Evaluación de costos

```bash
python3 evaluar_cgra.py --reporte reduccion=resultados/prueba_001/reduccion --reporte matmul=resultados/prueba_001/matmul --reporte convolucion=resultados/prueba_001/convolucion --salida resultados/prueba_001/evaluacion.json
```

No compila, mapea ni simula. Lee de cada carpeta los ciclos y NOPs de `reporte_ciclos.txt` y
la memoria final de `memoria.bin`, como `rl/medicion/simulacion.py`, y sigue el contrato con
la solución RL (`dijkstra/contrato.py`):

- **Validación previa**: las reglas de `rl/agente/paquete.py`. Si no las cumple, el estado es `accion_invalida`.
- **Validación funcional**: compara la memoria final con el grafo ejecutado sobre
  `cgra/programas/<operacion>/memoria.bin`, elemento por elemento. Si algún elemento supera
  ε_abs + ε_rel·|y|, es `incorrecta`.
- **C_A**: w = 4, 8, 1, 2 sobre las capacidades de cada PE.
- **C_L, por operación**: ciclos·(1 + NOPs/(ciclos·PEs)) (`ciclos_x_nops`).
- **C_R, por operación**: promedio de |y − ŷ|/(|y| + ε).
- **C**: promedio sobre las operaciones de K_A·C_A/C_A_ref + K_L·C_L,op/C_L_ref,op + K_R·C_R,op,
  si todas son `correcta`. Si no, C = P_fallo = 10.

Los normalizadores son los congelados de `config/normalizacion.json`, medidos con la
solución RL. Si una operación no está ahí, la evaluación de la referencia usa su propia
medición; las siguientes necesitan `--normalizadores resultados/greedy/normalizacion.json`.

Evalúe las tres operaciones con la misma configuración. No mezcle reportes de distintas mallas.

## 6. Siguiente ronda de Greedy

```bash
python3 agente_greedy.py --salida resultados/greedy --evaluaciones resultados/prueba_001/evaluacion.json
```

Registra cada evaluación como la solución RL, con los mismos formatos:

- `historial.jsonl`: solo se agregan líneas. Incluye estado, métricas, costo y la recompensa
  r = C_best − C (−P_fallo si se penaliza). Todas las de un lote se miden contra el mejor
  anterior al lote.
- `optimas.json`: las correctas empatadas en el menor costo (`tolerancia_empate_costo`).
- `resumen.json`: C_min, conteo por estado, porcentajes y evolución del costo.

Además actualiza la cadena de mejoras (`cadena.json`, `mejor_cgra.json`) y propone vecinas
de las mejores conocidas que pasan la validación previa. La primera evaluación debe ser la
de la referencia; si no es correcta, la búsqueda se detiene.

El presupuesto es el del RL (`config/agente.json`). `max_evaluaciones` no cuenta la
referencia. `paciencia` cuenta las evaluaciones seguidas sin mejora; `error_infraestructura`
no cuenta para ella. Las propuestas se recortan a las evaluaciones que quedan, y al agotarse
`resumen.json → estado` dice por qué terminó.

Puede proporcionar varios archivos después de `--evaluaciones`. Las propuestas todavía no
evaluadas se conservan. Los nombres `config_*.json` se renumeran al actualizar propuestas:
use carpetas distintas para los resultados de cada configuración y guarde una copia de su
JSON antes de avanzar a la siguiente ronda. Para iniciar otra búsqueda o cambiar
parámetros, use otra carpeta `--salida`.
