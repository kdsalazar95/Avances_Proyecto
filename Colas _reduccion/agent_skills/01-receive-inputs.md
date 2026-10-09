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
  "filas": 4,
  "columnas": 4,
  "palabras_memoria": 4096,
  "tipos_pe": {
    "M": {"nombre": "memoria", "instrucciones": ["NOP", "MOV", "LD", "ST", "SEND", "RECV"]},
    "C": {"nombre": "computo", "instrucciones": ["NOP", "MOV", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV"]}
  },
  "distribucion": {"oeste": "M", "este": "M", "resto": "C"}
}
```

1. Read `filas` and `columnas`: the mesh size. Whatever numbers they hold are
   the mesh for this run; never assume a size.
2. Get the type of every `PE{fila}{columna}`. The file gives it as a rule,
   `distribucion`, so the mesh follows `filas` and `columnas` without being
   rewritten: a PE on the `oeste`/`este`/`norte`/`sur` border takes that
   border's type (in a corner, `oeste`/`este` win) and every other PE takes
   `resto`. If the file has an explicit `malla` (one string per row, one letter
   per PE), it replaces the rule. Do not work it out by hand; print the map
   with the same code `run_cgra.py` uses:

   ```sh
   cd src && python3 -c "from formato_cgra import cargar_formato; print('\n'.join(cargar_formato().mapa()))"
   ```

   Read `tipos_pe`: for each letter, the instructions that type of PE can
   execute. A PE never executes an instruction outside its type;
   `run_cgra.py` refuses to load a program that does.
3. Classify every PE as a **memory PE** (its type has `LD`/`ST`) or a
   **compute PE** (its type has `ADD`/`MUL`). With the default file, the west
   and east columns are memory PEs and the rest are compute PEs:

   ```text
   M C C M      memoria (M): LD, ST, SEND, RECV, MOV, NOP   -> moves data, never computes
   M C C M      computo (C): ADD, SUB, MUL, DIV, SEND, RECV, MOV, NOP
   M C C M                                                   -> computes, never touches memory
   M C C M
   ```

4. Read `palabras_memoria`: the size of the data memory, in words. The
   `memoria.bin` of the program must fit in it.
5. Treat those values as authoritative and continue without asking, even when
   the mesh does not match the size of the program: a larger mesh leaves its
   extra PEs in `NOP` and a smaller one is handled with tiles, so there is
   nothing to decide. Only a different CGRA that the user states explicitly in
   the same request overrides the file. State in one line which mesh and which
   type map were used, and what that implied.
6. If the file is missing or invalid, or a row has no memory PE (its compute PEs
   could never receive data), stop and ask the user.

The CGRA is not told which operation it runs, the same way a processor is not:
`run_cgra.py` has no operation setting. The pipeline plays the role of the
compiler, so the program to compile is the `.c` source named in the request.

## Finding the Source

Take the source from the request, either as a path (`src/matmul/matmul.c`) or
as the program's folder name (`matmul`). Each program has its own folder, with
the C source inside:

| Program | Source | Size constants |
|---|---|---|
| `reduccion` | `src/reduccion/reduccion.c` | `TAMANO_VECTOR` |
| `matmul` | `src/matmul/matmul.c` | `TAMANO_MATRIZ` |
| `convolucion` | `src/convolucion/convolucion.c` | `TAMANO_IMAGEN`, `TAMANO_KERNEL` |

If the request does not name a source, ask which one to compile. A new program
follows the same layout, `src/<programa>/<programa>.c`, and must define an
`inicializar_memoria()` function (see [Generate LLVM IR](02-generate-llvm-ir.md)).

Read the sizes from the `#define` lines of that source; they may carry a
trailing comment. `convolucion.c` also defines `TAMANO_SALIDA` as an expression,
so compute it as `TAMANO_IMAGEN - TAMANO_KERNEL + 1`.

The output size follows from those constants, and it is also the shape of the
`result` region that `inicializar_memoria()` writes to `memoria.bin`:

```text
reduccion   -> one value, result[0]
matmul      -> TAMANO_MATRIZ x TAMANO_MATRIZ
convolucion -> TAMANO_SALIDA x TAMANO_SALIDA
```

## Mesh Size vs Problem Size

The mesh and the problem do **not** have to match:

- **Problem larger than the mesh:** keep the mesh and process the work in tiles
  of `filas x columnas` (skill 06). Never invent extra PEs.
- **Mesh larger than the problem:** the PEs with no work stay idle with `NOP`.
  In a reduction the idle compute PEs still start with `MOV acc, 0.0` and take
  part in the reduction route, so the route does not change.

## Actions

1. Read the CGRA from `src/cgra.json` and locate the source named in the
   request.
2. Record its `#define` values and the resulting output size.
3. Set `output_dir` to the program's folder, `src/<programa>/`. The IR, the
   graphs, and `memoria.bin` are generated there, and the PE files go to
   `src/<programa>/pe_instructions/` (skill 08).
4. Check whether the `.ll`, the two `.dot` graphs, and `memoria.bin` already
   exist in that folder, and treat missing or stale ones as rebuildable
   artifacts.
5. Record the mesh size and the type map read from `cgra.json`. Pass them to
   every later skill; do not hard-code a size or a type map downstream.
6. Only compute PEs own work. The tile of skill 06 is therefore the block of
   compute PEs: with the default file, `tile_filas = filas` and
   `tile_columnas = columnas - 2` (the compute columns). A tile is never larger
   than that block.

## Outputs

Return a normalized input manifest:

```text
programa=<folder name of the source>
source_c=src/<programa>/<programa>.c
parametros=<#define values, e.g. TAMANO_MATRIZ=8>
llvm_ir=<path or missing>
cfg_dot=<path or missing>
ddg_dot=<path or missing>
memoria=src/<programa>/memoria.bin (or missing)
output_dir=src/<programa>/
cgra=src/cgra.json
mesh_filas=<int, from cgra.json>
mesh_columnas=<int, from cgra.json>
palabras_memoria=<int, from cgra.json>
malla=<one row per line, e.g. M C C M>
memory_pes=<PE ids of type memoria>
compute_pes=<PE ids of type computo>
tile_filas=<int, rows of the compute block>
tile_columnas=<int, columns of the compute block>
salida_filas=<int, from the size constants>
salida_columnas=<int, from the size constants>
```

## Notes

Do not assume the existing `.ll`, graph, or `memoria.bin` files are current. If
a `#define` or the source changed since they were produced, regenerate them in
skills 02 and 03, and regenerate the PE files too: a schedule is only valid for
the sizes it was built from.
