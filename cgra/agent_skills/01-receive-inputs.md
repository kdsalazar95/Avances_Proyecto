# Skill: Receive Inputs

## Purpose

Collect the configuration, source, IR, graph, and memory inputs that the rest of
the pipeline needs. Prefer the program's `.c` source as the source of truth when
files disagree.

## Reading the Configuration

`src/cgra.json` describes the CGRA, so it does not have to be repeated in every
request. The same file is read by `run_cgra.py` (`src/formato_cgra.py`), so the
compiler and the simulator always target the same hardware. The CGRA has no
fixed size: build it from this file on every run. A file looks like this (the
size is an example):

```json
{
  "filas": <filas>,
  "columnas": <columnas>,
  "palabras_memoria": <palabras>,
  "ancho_vector": <W>,
  "tipos_pe": {
    "M": {"nombre": "memoria",   "instrucciones": ["NOP", "MOV", "LD", "ST", "SEND", "RECV", "VLD", "VST", "VSEND", "VRECV"]},
    "E": {"nombre": "escalar",   "instrucciones": ["NOP", "MOV", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV"]},
    "V": {"nombre": "vectorial", "instrucciones": ["NOP", "MOV", "SEND", "RECV", "VADD", "...", "VSEND", "VRECV"]},
    "C": {"nombre": "computo",   "instrucciones": ["NOP", "MOV", "ADD", "...", "VADD", "...", "VSEND", "VRECV"]}
  },
  "distribucion": {
    "ciclo_filas": [
      {"oeste": "M", "este": "M", "resto": "E"},
      {"oeste": "M", "este": "M", "resto": "V"},
      {"oeste": "M", "este": "M", "resto": "C"}
    ]
  }
}
```

`ancho_vector` is optional: without it the CGRA has no vector registers.

1. Read `filas` and `columnas`: the mesh size (`filas x columnas`). Whatever
   numbers they hold are the mesh for this run; never assume a size.
2. Get the type of every `PE{fila}{columna}`. The file gives it as a rule,
   `distribucion`, so the mesh follows `filas` and `columnas` without being
   rewritten: a PE on the `oeste`/`este`/`norte`/`sur` border takes that
   border's type (in a corner, `oeste`/`este` win) and every other PE takes
   `resto`. If the file has an explicit `malla` (one string per row, one letter
   per PE), it replaces the rule. Do not work it out by hand; print the map and
   the instructions of each type with the same code `run_cgra.py` uses:

   ```sh
   cd src && python3 -c "from formato_cgra import FormatoCGRA; f = FormatoCGRA.cargar(); print('\n'.join(f.mapa() + f.leyenda()))"
   ```

   A PE never executes an instruction outside its type; `run_cgra.py` refuses
   to load a program that does.
3. Classify every PE as a **memory PE** (its type has `LD`/`ST`) or a
   **compute PE** (its type has arithmetic). The west and east columns are
   only memory PEs (`M`), which read and write the data memory, scalars and
   vectors, and send data to their neighbors; every other PE is a compute PE,
   which only executes instructions and never touches memory. A compute PE is
   **scalar** if its type has `ADD`..`DIV`, **vector** if it has
   `VADD`..`VBCAST`, or both. With `ciclo_filas` the rows repeat its rules in
   order (row `f` takes rule `f mod` their count); the default rotates
   `escalar`, `vectorial`, `computo`:

   ```text
   memory PEs:   PE(f, 0) and PE(f, columnas - 1), for every row f
   compute PEs:  PE(f, c) with 1 <= c <= columnas - 2
   f mod 3 = 0:  M E E ... E M
   f mod 3 = 1:  M V V ... V M
   f mod 3 = 2:  M C C ... C M

   M (memoria):   NOP, MOV, LD, ST, VLD, VST, SEND, RECV, VSEND, VRECV  -> moves data, never computes
   E (escalar):   NOP, MOV, ADD, SUB, MUL, DIV, SEND, RECV             -> scalar arithmetic only
   V (vectorial): NOP, MOV, SEND, RECV, VADD, VSUB, VMUL, VDIV,
                  VREDSUM, VBCAST, VSEND, VRECV                        -> vector arithmetic only
   C (computo):   E + V                                                -> scalar and vector
   ```

   `V` keeps `SEND`/`RECV` only to move the scalars of `VREDSUM`/`VBCAST`
   and to relay values; it never adds scalars (skills 06 and 07 say how a `V`
   row takes part in a reduction). `distribucion` may also be
   `filas_pares` / `filas_impares` (two alternating rules) or a single rule
   (every row alike).

   Use the map printed by the command above as the picture of the CGRA for this
   run; the later skills apply their rules to it.

   If the file has no `tipos_pe`, every PE is generic and runs the whole ISA.
   Plan with the same layout anyway (memory PEs on the west and east columns):
   a generic PE accepts both roles, so the same skills apply.
4. Read `palabras_memoria`: the size of the data memory, in words. The
   `memoria.bin` of the program must fit in it.
5. Decide the **mode** of the run. It is `vectorial` when the file has
   `ancho_vector` (`W`, the words of every vector register) **and** the map has
   vector compute PEs (`V` or `C`) next to memory PEs, with the vector
   instructions the vector schedules use:

   ```text
   memory PEs (M):          VLD, VST, VSEND, VRECV
   vector compute PEs:      VADD, VMUL, VREDSUM, VBCAST, VSEND, VRECV
   ```

   Otherwise (no `ancho_vector`, or a mesh of one row, which only has the
   `escalar` row 0) the mode is `escalar`, and the vector instructions are never
   emitted. In `vectorial` mode the vector compute PEs (`V`, `C`) do the work in
   blocks of `W` contiguous elements, and the scalar compute PEs (`E`) do, at
   the same time,
   the elements that do not fill a block (skills 05-07). State the mode, and `W`, in the same line as the
   mesh.
6. Treat those values as authoritative and continue without asking, even when
   the mesh does not match the size of the program: a larger mesh leaves its
   extra PEs in `NOP` and a smaller one is handled with tiles, so there is
   nothing to decide. Only a different CGRA that the user states explicitly in
   the same request overrides the file. State in one line which mesh and which
   type map were used, and what that implied.
7. If the file is missing or invalid, or the mesh has fewer than 3 columns (no
   room for a compute PE between the two memory columns), stop and ask the
   user.

The CGRA is not told which operation it runs, the same way a processor is not:
`run_cgra.py` has no operation setting. The pipeline plays the role of the
compiler, and it always compiles **every operation** of `operaciones.c` (see
[All Operations](#all-operations)).

## Finding the Source

Every operation lives in one source, `operaciones/operaciones.c`, as a pair of
functions: `<operacion>()`, the work to schedule, and `inicializar_<operacion>()`,
which saves its initial memory:

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

## All Operations

Every run of the pipeline compiles **all the operations**: every entry of the
`OPERACIONES` table at the end of `operaciones/operaciones.c`, in table order.
Build that list from the table, not from the request, and do not ask which
operations to compile. A request that names a single operation does not change
this: compile all of them anyway.

Read `src/cgra.json` once: every operation targets the same CGRA. Then compile
the operations **one at a time**: build the manifest of the
first operation (see [Outputs](#outputs)), run skills 02 to 09 for it, and only
then start the next one. Each operation writes only to its own
`programas/<operacion>/`, so compiling one never touches the files of another.

If a skill fails for one operation, record the error, leave that operation's
folder as it is, and continue with the next operation. Do not stop the whole
run because one operation failed: every operation of the table must be
attempted.

## Mesh Size vs Problem Size

The mesh and the problem do **not** have to match:

- **Problem larger than the mesh:** keep the mesh and process the work in tiles
  of the compute block (skill 06). Never invent extra PEs.
- **Mesh larger than the problem:** the PEs with no work stay idle with `NOP`.
  In a reduction the idle compute PEs still start with `MOV acc, 0.0` and take
  part in the reduction route, so the route does not change.

## Actions

1. Read the CGRA from `src/cgra.json`, build the list of all the operations from
   the `OPERACIONES` table, and locate each one in `operaciones/operaciones.c`. The remaining
   actions apply to one operation at a time.
2. Record its `#define` values and the resulting output size.
3. Set `output_dir` to the operation's folder, `programas/<operacion>/`. The
   IR, the graphs, and `memoria.bin` are generated there, and the PE files go
   to `programas/<operacion>/pe_instructions/` (skill 08).
4. Check whether the `.ll`, the two `.dot` graphs, `memoria.bin` and
   `pe_instructions/` already exist. They are generated artifacts: regenerate
   them on every invocation from the current source and CGRA configuration.
   Existing files, missing PEs or a mismatch with the previous mesh are reasons
   to rebuild, never reasons to stop or ask for confirmation. Skill 08 overwrites
   every PE file and removes obsolete generated PE files.
5. Record the mesh size and the type map read from `cgra.json`. Pass them to
   every later skill; do not hard-code a size or a type map downstream.
6. Only compute PEs own work. The tile of skill 06 is therefore the **compute
   block**, the columns between the two memory columns:
   `tile_filas = filas` and `tile_columnas = columnas - 2`.

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
malla=<one row per line, e.g. M E E M>
memory_pes=<PE ids of the memory PEs (M)>
compute_pes=<PE ids of the compute PEs (E, V and C)>
vector_compute_pes=<PE ids of the vector compute PEs (V and C), row-major>
scalar_compute_pes=<PE ids of the scalar-only compute PEs (E), row-major>
tile_filas=<int, filas>
tile_columnas=<int, columnas - 2>
modo=<escalar or vectorial>
ancho_vector=<W from cgra.json, or none in escalar mode>
salida_filas=<int, from the size constants>
salida_columnas=<int, from the size constants>
```

Finish with one summary line per operation of the table, after skill 09 of the
last one:

```text
operacion     estado   detalle
reduccion     ok       programas/reduccion/pe_instructions/ (<PEs> PEs, <N> ciclos)
matmul        ok       programas/matmul/pe_instructions/ (<PEs> PEs, <N> ciclos)
convolucion   error    skill <NN>: <the error, in one line>
```

## Notes

Do not assume existing artifacts are current. Rebuild them in skills 02 to 08
and validate them in skill 09 on every invocation. A change to a `#define`, the
operation's code, or `cgra.json` (mesh size, distribution or explicit map, PE
instruction lists, vector width, or memory capacity) invalidates the previous
schedule. The current source defines the computation and initial memory;
`cgra.json` defines the hardware it must run on.
