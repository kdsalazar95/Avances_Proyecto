# Agent Skill Pipeline: Mesh Scheduler

Use these skills in order to transform one of the supported C sources into
per-PE instruction files for a `filas x columnas` processing-element mesh, plus
the initial memory those programs run on (`memoria.bin`).

The supported sources are `src/reduccion/reduccion.c`, `src/matmul/matmul.c`,
and `src/convolucion/convolucion.c`. Their sizes come from the `#define`
constants of each source, the source to compile comes from the request, and the
CGRA (mesh size, memory size and the type of every PE) comes from
`src/cgra.json` (see [Receive Inputs](01-receive-inputs.md)).

The CGRA behaves like a processor: it is never told which operation it runs.
The pipeline is the compiler. It turns the source into one program per PE
(skill 08), and running the source once produces the initial memory:
`inicializar_memoria()` in each `.c` fills the arrays and saves them to
`src/<programa>/memoria.bin` (skill 02). `run_cgra.py` only loads those two
things and executes them.

The mesh and the problem do not have to be the same size: a larger problem is
processed in tiles of the mesh size, and a larger mesh leaves its extra PEs in
`NOP`.

## Pipeline

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

```sh
clang-18 -S -emit-llvm <programa>.c -o <programa>.ll
clang-18 <programa>.c -o <programa>.out && ./<programa>.out   # writes memoria.bin
opt-18 -passes=dot-cfg <programa>.ll
opt-18 -passes=dot-ddg <programa>.ll
```

## Default CGRA (`src/cgra.json`)

`4x4` mesh, one `PE{fila}{columna}` per position, connected only to its north,
south, east, and west neighbors. The west and east columns are **memory PEs**
and the rest are **compute PEs**:

```text
PE00 (M) - PE01 (C) - PE02 (C) - PE03 (M)
PE10 (M) - PE11 (C) - PE12 (C) - PE13 (M)
PE20 (M) - PE21 (C) - PE22 (C) - PE23 (M)
PE30 (M) - PE31 (C) - PE32 (C) - PE33 (M)
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
