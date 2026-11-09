# Archivos `.ll` y `.dot`, y qué toma el bloque Dijkstra

Referencia de los archivos que dejan los skills 02 y 03 en `cgra/programas/<operacion>/`, de lo que contiene cada uno y de cuáles usa el bloque Dijkstra (`dijkstra_greedy/dijkstra/`).

## Resumen

| | `operaciones.ll` | `.<operacion>.dot` (CFG) | `ddg.<operacion>..dot` (DDG) |
|---|---|---|---|
| Qué es | El programa en el IR de LLVM, en texto | Grafo de control de una función, para dibujar | Grafo de dependencias de una función, para dibujar |
| Lo genera | `clang-18 -S -emit-llvm` (skill 02) | `opt-18 -passes=dot-cfg` a partir del `.ll` (skill 03) | `opt-18 -passes=dot-ddg` a partir del `.ll` (skill 03) |
| Nodos | Instrucciones agrupadas en bloques | Bloques básicos (con sus instrucciones como etiqueta) | Instrucciones (o grupos de instrucciones) |
| Aristas | Saltos (`br`) y uso de valores (`%25`) | Saltos posibles entre bloques | "Esta instrucción usa el resultado de aquella" |
| Bucles | Con sus límites (`icmp slt i32 %20, 9`) | Un ciclo en el dibujo, sin cantidad de vueltas | El cuerpo una sola vez, con flechas de vuelta |
| Índices concretos (`a[2][5]`) | Se pueden calcular | No | No |
| Se puede ejecutar | Sí | No | No |
| Uso | Compilar, analizar, ejecutar | Mirar la estructura: `dot -Tpng .matmul.dot -o cfg.png` | Mirar las dependencias: `dot -Tpng ddg.matmul..dot -o ddg.png` |
| **Lo usa Dijkstra** | **Sí** | No | No |

Los `.dot` salen del `.ll`, así que no tienen información que el `.ll` no tenga. Son una vista para personas.

## `operaciones.ll` (skill 02)

```sh
clang-18 -S -emit-llvm ../../operaciones/operaciones.c -o operaciones.ll
```

Es `operaciones.c` traducido a instrucciones simples, sin optimizar (-O0). Tiene todas las funciones del `.c`: `main`, las tres operaciones, sus `inicializar_*`, `guardar_memoria` y `escribir`.

Fragmento de `@matmul`:

```llvm
18:                                       ; inicio de C[i][j]
  store float 0.000000e+00, ptr %6        ; acc = 0
  br label %19
19:
  %20 = load i32, ptr %7                  ; k
  %21 = icmp slt i32 %20, 9               ; ¿k < 9?
  br i1 %21, label %22, label %42         ; sí: cuerpo del bucle; no: guardar el resultado
22:
  %25 = getelementptr inbounds [9 x [9 x float]], ptr %1, i64 0, i64 %24   ; dirección de a[i]
  ...
  %38 = call float @llvm.fmuladd.f32(float %29, float %36, float %37)     ; acc += a * b
```

Qué datos tiene:

- **Variables locales y arreglos**: cada una es un `alloca` (`%1 = alloca [9 x [9 x float]]`). Con -O0 los contadores `i`, `j`, `k` y `acc` también viven en memoria (`load`/`store` a su `alloca`).
- **Tamaños**: en los tipos (`[9 x [9 x float]]`) y en las comparaciones de los bucles (`icmp slt ..., 9`).
- **Direcciones**: `getelementptr` calcula la posición de cada acceso a partir de los índices.
- **Aritmética**: `fadd`, `fmul` y `llvm.fmuladd` (clang une `acc += x * y` en una sola llamada; el bloque la separa en MUL + ADD).
- **Qué arreglo es cada región de `memoria.bin`**: la tabla `RegionMemoria` que arma `@inicializar_<operacion>`, con el nombre (`@.str.N = "a"`) y el parámetro que guarda.
- **Sin nombres de valores**: el clang de la distribución los descarta (`%25`, no `%a_ik`). Por eso el bloque identifica los arreglos por la tabla de regiones y no por el nombre.

## `.<operacion>.dot`, CFG (skill 03)

```sh
opt-18 -passes=dot-cfg -disable-output operaciones.ll
```

Un archivo por función (`.matmul.dot`, `.main.dot`, …). Cada nodo es un bloque del `.ll` con sus instrucciones como texto, y cada flecha, un salto posible con su probabilidad estimada:

```
Node0x64a374049ab0 [shape=record, ..., label="{0:\l| %1 = alloca [9 x [9 x float]], align 16\l ...}"];
Node0x64a374049ab0 -> Node0x64a37404a310 [tooltip="0 -> 11\nProbability 100.00%"];
```

Sirve para ver la forma de los bucles (cuáles hay y cómo se anidan). No dice cuántas veces se repite cada uno.

## `ddg.<operacion>..dot`, DDG (skill 03)

```sh
opt-18 -passes=dot-ddg -disable-output operaciones.ll
```

Un archivo por función (`ddg.matmul..dot`, …). Cada nodo es una instrucción (o un grupo) y cada flecha, una dependencia: `[def-use]` (una instrucción usa el valor de otra) o `[rooted]` (arista desde un nodo raíz que agrupa el grafo). En `ddg.matmul..dot` hay 25 `[def-use]` y 12 `[rooted]`:

```
Node0x59e85cc9ffb0 -> Node0x59e85cc9f0e0 [label="[rooted]"];
```

Muestra el cuerpo de cada bucle **una sola vez**. Con -O0 también trae como dependencias los `load`/`store` de los contadores y de `acc`, que no son cómputo real.

## Por qué Dijkstra lee el `.ll` y no los `.dot`

Dijkstra necesita el grafo **desenrollado**: cada operación por separado, con su dirección de memoria concreta (matmul: 1458 loads, 729 MUL, 729 ADD, 81 constantes y 81 stores = 3078 nodos). Para armarlo hacen falta los límites de los bucles y el cálculo de cada índice, que solo están en el `.ll`:

1. El DDG tiene el cuerpo del bucle una vez, no sus 729 repeticiones.
2. A -O0 está lleno de dependencias de contadores y de `acc` que habría que limpiar, y para eso igual hace falta el IR.
3. El `.dot` es texto para dibujar: leerlo obliga a interpretar instrucciones LLVM escritas como etiquetas, que es lo mismo que leer el `.ll`, pero en un formato peor.

Por eso `ir_llvm.py` **ejecuta simbólicamente** `@<operacion>` del `.ll`. Los enteros (índices, contadores) se calculan de verdad, y cada `load`/`store`/`fadd`/`fmul` de float se convierte en un nodo del grafo. El skill 03 queda como ayuda para mirar la estructura.

## Qué toma el bloque Dijkstra (sus entradas)

| Entrada | De dónde | Qué usa de ahí | Para qué |
|---|---|---|---|
| `operaciones.ll` | `programas/<operacion>/` (skill 02) | `@<operacion>` (el cálculo), `@inicializar_<operacion>` (la tabla de regiones) y las cadenas `@.str.N` | Armar el grafo desenrollado (skill 04) |
| `memoria.bin` | `programas/<operacion>/` (skill 02) | Para el mapeo, **solo la tabla de símbolos**: nombre, filas y columnas de cada región | Saber qué región es cada arreglo y comprobar su tamaño |
| Configuración de la CGRA | `cgra.json` + tamaño (`--filas`/`--columnas`), o un `.json` con tamaño fijo y `malla` | `filas`, `columnas`, `tipos_pe` con sus `instrucciones`, `malla` o `distribucion`, `ancho_vector` | Armar la malla, decidir qué PE ejecuta qué y agrupar en vectores de `ancho_vector` (skills 05–06) |
| `compatibilidad.json` | `dijkstra_greedy/config/` | Operación del grafo → instrucción (`fadd` → `ADD`, `vload` → `VLD`…) y las instrucciones de transferencia (`SEND`/`RECV`, `VSEND`/`VRECV`) | Candidatos de cada operación y rutas |
| Registros del ISA | `cgra/src/isa.py` | 5 escalares (`rA rB rC rT acc`) y 8 vectoriales (`v0`–`v7`) | Límite de valores vivos por PE y nombres en los `PE*.txt` |
| Opciones | `Opciones` y `modo` | `reenvio_memoria`, `holgura`, `intentos_huecos`, `ventana`, `max_candidatos`, `escalar`/`vectorial`/`auto`/`mejor` | Ajustes del algoritmo |

**No toma**:
- los `.dot` (skill 03);
- `operaciones.c` directamente (lo usa a través del `.ll`);
- los `pe_instructions/`, `manifest.json` ni reportes del flujo con LLM;
- los valores de `memoria.bin` para mapear: el mapeo no depende de los datos.

**Después del mapeo**, el evaluador sí usa los **valores** de `memoria.bin`: los carga en el simulador y calcula el resultado esperado (C_R). `palabras_memoria` de la configuración solo lo usa el cargador del simulador.

**Salida del bloque** (`ResultadoMapeo`):
- `valido` y `motivo`;
- la ubicación de cada operación (PE y ciclo);
- el programa de cada PE, una instrucción por ciclo, que se escribe como `PE*.txt`;
- las métricas: ciclos, NOPs, slots, ratio de NOPs, transferencias y modo.
