# Agent Skill Pipeline: Mesh Scheduler

Use these skills in order to transform one of the supported C sources into
per-PE instruction files for a `filas x columnas` processing-element mesh, plus
the initial memory those programs run on (`memoria.bin`).

The supported operations are `reduccion`, `matmul` and `convolucion`, all in
one source, `operaciones/operaciones.c`. Their sizes come from the `#define`
constants of each operation, the operations to compile come from the request
(one, several, or all of them), and the mesh size comes from `filas` and `columnas` in `src/cgra.json`, the same file
`run_cgra.py` reads (see [Receive Inputs](01-receive-inputs.md)).

The CGRA behaves like a processor: it is never told which operation it runs.
The pipeline is the compiler. It turns the source into one program per PE
(skill 08), and running the source once produces the initial memory:
`inicializar_<operacion>()` fills the operation's arrays and saves them to
`programas/<operacion>/memoria.bin` (skill 02). `run_cgra.py` only loads those two
things and executes them.

The mesh and the problem do not have to be the same size: a larger problem is
processed in tiles of the mesh size, and a larger mesh leaves its extra PEs in
`NOP`.

## Pipeline

Skills 02 to 09 compile **one operation**. When a request names several
operations, or all of them, skill 01 reads the CGRA once and then the pipeline
runs from skill 02 to skill 09 once per operation, one after the other, each in
its own `programas/<operacion>/`. A failure in one operation does not stop the
others, and the run ends with one summary line per operation (see
[Receive Inputs](01-receive-inputs.md#one-several-or-all-operations)).

1. [Receive Inputs](01-receive-inputs.md)
2. [Generate LLVM IR and Initial Memory](02-generate-llvm-ir.md)
3. [Generate Graphs](03-generate-graphs.md)
4. [Parse Program](04-parse-program.md)
5. [Detect Parallelism](05-detect-parallelism.md)
6. [Map to Mesh](06-map-to-mesh.md)
7. [Schedule Instructions](07-schedule-instructions.md)
8. [Generate PE Files](08-generate-pe-files.md)
9. [Validate Schedule](09-validate-schedule.md)

## Default Commands

From `programas/<operacion>/`:

```sh
clang-18 -S -emit-llvm ../../operaciones/operaciones.c -o operaciones.ll
clang-18 ../../operaciones/operaciones.c -o operaciones.out && ./operaciones.out <operacion>   # writes memoria.bin
opt-18 -passes=dot-cfg -disable-output operaciones.ll
opt-18 -passes=dot-ddg -disable-output operaciones.ll
```

## Default Mesh

`2x2` example; the mesh grows to any `filas x columnas` by extending the same
grid, one `PE{fila}{columna}` per position, connected only to its north, south,
east, and west neighbors:

```text
PE00 --east/west-- PE01
 |                 |
north/south        north/south
 |                 |
PE10 --east/west-- PE11
```

## Default Instruction Set

```text
MOV  reg, imm
LD   reg, mem[index]
ST   mem[index], reg
ADD  dst, src1, src2
MUL  dst, src1, src2
SEND dir, reg
RECV dir, reg
NOP
```
