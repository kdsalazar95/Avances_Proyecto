# Skill: Receive Inputs

## Purpose

Collect the configuration, source, IR, graph, and memory inputs that the rest of
the pipeline needs. Prefer the program's `.c` source as the source of truth when
files disagree.

## Reading the Configuration

`src/cgra.json` describes the CGRA, so it does not have to be repeated in every
request. The same file is read by `run_cgra.py` (`src/formato_cgra.py`), so the
compiler and the simulator always target the same hardware:

```json
{
  "filas": <filas>,
  "columnas": <columnas>,
  "palabras_memoria": <palabras>
}
```

All PEs are identical (homogeneous CGRA): every PE can execute every
instruction of the ISA, so any PE may load, store, compute and communicate.

1. Read `filas` and `columnas` from that file. They are the mesh size
   (`filas x columnas`). Whatever numbers they hold are the mesh for this run;
   never assume a size.
2. Read `palabras_memoria`: the size of the data memory, in words. The
   `memoria.bin` of the program must fit in it.
3. Treat those values as authoritative and continue without asking, even when
   the mesh does not match the size of the program: a larger mesh leaves its
   extra PEs in `NOP` and a smaller one is handled with tiles, so there is
   nothing to decide. Only a different mesh that the user states explicitly in
   the same request overrides the file. State in one line which mesh was used,
   and what that implied.
4. If the file is missing or invalid, ask the user for the mesh size.

The CGRA is not told which operation it runs, the same way a processor is not:
`run_cgra.py` has no operation setting. The pipeline plays the role of the
compiler, so the operations to compile are the ones named in the request.

## Finding the Source

Every operation lives in one source, `operaciones/operaciones.c`, as a pair of
functions: `<operacion>()`, the work to schedule, and `inicializar_<operacion>()`,
which saves its initial memory. Take the operation's name from the request
(`matmul`):

| Operation | Function | Size constants |
|---|---|---|
| `reduccion` | `reduccion()` | `TAMANO_VECTOR` |
| `matmul` | `matmul()` | `TAMANO_MATRIZ` |
| `convolucion` | `convolucion()` | `TAMANO_IMAGEN`, `TAMANO_KERNEL` |

The available names are the ones in the `OPERACIONES` table at the end of
`operaciones.c`. A new operation follows the same pattern: its two functions
plus an entry in that table (see [Generate LLVM IR](02-generate-llvm-ir.md)).

Read the sizes from the `#define` lines of the operation's section of
`operaciones.c`; they may carry a trailing comment. `convolucion` also defines
`TAMANO_SALIDA` as an expression, so compute it as
`TAMANO_IMAGEN - TAMANO_KERNEL + 1`.

The output size follows from those constants, and it is also the shape of the
`result` region that `inicializar_<operacion>()` writes to `memoria.bin`:

```text
reduccion   -> one value, result[0]
matmul      -> TAMANO_MATRIZ x TAMANO_MATRIZ
convolucion -> TAMANO_SALIDA x TAMANO_SALIDA
```

## One, Several, or All Operations

A request can name one operation (`matmul`), several (`reduccion y matmul`), or
all of them ("todas las operaciones", "all operations"):

1. Build the list of operations to compile:
   - the ones named, in the order they were named;
   - for "all", every entry of the `OPERACIONES` table, in table order.
2. Reject a name that is not in the table, and say which names exist. Compile
   the valid ones anyway.
3. If the request names no operation and does not say "all", ask which ones to
   compile.

Read `src/cgra.json` once: every operation of the request targets the same
CGRA. Then compile the operations **one at a time**: build the manifest of the
first operation (see [Outputs](#outputs)), run skills 02 to 09 for it, and only
then start the next one. Each operation writes only to its own
`programas/<operacion>/`, so compiling one never touches the files of another.

If a skill fails for one operation, record the error, leave that operation's
folder as it is, and continue with the next operation. Do not stop the whole
request because one operation failed.

## Mesh Size vs Problem Size

The mesh and the problem do **not** have to match:

- **Problem larger than the mesh:** keep the mesh and process the work in tiles
  of `filas x columnas` (skill 06). Never invent extra PEs.
- **Mesh larger than the problem:** the PEs with no work stay idle with `NOP`.
  In a reduction they still start with `MOV acc, 0.0` and take part in the
  reduction route, so the route does not change.

## Actions

1. Read the mesh size from `src/cgra.json`, build the list of operations of the
   request, and locate each one in `operaciones/operaciones.c`. The remaining
   actions apply to one operation at a time.
2. Record its `#define` values and the resulting output size.
3. Set `output_dir` to the operation's folder, `programas/<operacion>/`. The
   IR, the graphs, and `memoria.bin` are generated there, and the PE files go
   to `programas/<operacion>/pe_instructions/` (skill 08).
4. Check whether the `.ll`, the two `.dot` graphs, and `memoria.bin` already
   exist in that folder, and treat missing or stale ones as rebuildable
   artifacts.
5. Record the mesh size read from `cgra.json`. Pass it to every later skill;
   do not hard-code `2x2` downstream.
6. Set `tile_filas = mesh_filas` and `tile_columnas = mesh_columnas`. A tile is
   never larger than the mesh.

## Outputs

Return one normalized input manifest per operation, and pass it to skills 02 to
09 of that operation:

```text
operacion=<name of the operation, e.g. matmul>
source_c=operaciones/operaciones.c
funcion=<operacion>
parametros=<#define values, e.g. TAMANO_MATRIZ=8>
llvm_ir=<path or missing>
cfg_dot=<path or missing>
ddg_dot=<path or missing>
memoria=programas/<operacion>/memoria.bin (or missing)
output_dir=programas/<operacion>/
cgra=src/cgra.json
mesh_filas=<int, from cgra.json>
mesh_columnas=<int, from cgra.json>
palabras_memoria=<int, from cgra.json>
tile_filas=<int, mesh_filas by default>
tile_columnas=<int, mesh_columnas by default>
salida_filas=<int, from the size constants>
salida_columnas=<int, from the size constants>
```

When the request has several operations, finish with one summary line per
operation, after skill 09 of the last one:

```text
operacion     estado   detalle
reduccion     ok       programas/reduccion/pe_instructions/ (<PEs> PEs, <N> ciclos)
matmul        ok       programas/matmul/pe_instructions/ (<PEs> PEs, <N> ciclos)
convolucion   error    skill <NN>: <the error, in one line>
```

## Notes

Do not assume the existing `.ll`, graph, or `memoria.bin` files are current. If
a `#define` or the operation's code changed since they were produced, regenerate them in
skills 02 and 03, and regenerate the PE files too: a schedule is only valid for
the sizes it was built from.
