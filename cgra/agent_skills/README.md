# Agent Skill Pipeline: Mesh Scheduler

Use these skills in order to transform every operation of the supported C
source into per-PE instruction files for a `filas x columnas` processing-element mesh, plus
the initial memory those programs run on (`memoria.bin`).

The supported operations are `reduccion`, `matmul` and `convolucion`, all in
one source, `operaciones/operaciones.c`. Their sizes come from the `#define`
constants of each operation, every run compiles all of them (the
`OPERACIONES` table of `operaciones.c`), and the CGRA (mesh size, memory size and the
type of every PE) comes from `src/cgra.json`, the same file `run_cgra.py` reads
(see [Receive Inputs](01-receive-inputs.md)).

The CGRA behaves like a processor: it is never told which operation it runs.
The pipeline is the compiler. It turns the source into one program per PE
(skill 08), and running the source once produces the initial memory:
`inicializar_<operacion>()` fills the operation's arrays and saves them to
`programas/<operacion>/memoria.bin` (skill 02). `run_cgra.py` only loads those two
things and executes them.

The mesh and the problem do not have to be the same size: a larger problem is
processed in tiles of the compute block, and a larger mesh leaves its extra PEs
in `NOP`. The size is not fixed: every skill builds the CGRA from
`src/cgra.json`.

## Pipeline

The pipeline always compiles **all the operations** of the `OPERACIONES` table,
never only one. Skill 01 reads the CGRA once and builds the list of operations;
then skills 02 to 09 run once per operation, one after the other, each in its
own `programas/<operacion>/`. A failure in one operation does not stop the
others, and the run ends with one summary line per operation (see
[Receive Inputs](01-receive-inputs.md#all-operations)).

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

## PE Types (`src/cgra.json`)

One `PE{fila}{columna}` per position, connected only to its north, south, east,
and west neighbors. The west and east columns are **memory PEs** and the rest
are **compute PEs**, whatever the size:

```text
memory PEs:   PE(f, 0) and PE(f, columnas - 1), for every row f
compute PEs:  PE(f, c) with 1 <= c <= columnas - 2
```

## Instruction Set per PE Type

```text
                       memoria (M)   computo (C)
MOV  reg, imm               x             x
LD   reg, mem[index]        x
ST   mem[index], reg        x
ADD  dst, src1, src2                      x
SUB  dst, src1, src2                      x
MUL  dst, src1, src2                      x
DIV  dst, src1, src2                      x
SEND dir, reg               x             x
RECV dir, reg               x             x
NOP                         x             x
```

Memory PEs load operands and send them to the compute PEs; compute PEs compute
and send their results back to be stored (skills 05-07). `run_cgra.py` refuses
to load a program with an instruction outside the type of its PE.
