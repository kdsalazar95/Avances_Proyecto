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

Apply the [Mesh Compatibility Contract](mesh-compatibility.md) throughout the
pipeline. It defines automatic small layouts (singleton G, vertical M/C,
two-column M C, and one/two-row meshes with M at both ends and C interiors),
capability checks, compatible routes, scalar remainders
and tiling. These are the requirements for the upcoming shared map resolver
and scheduling implementation. This documentation update does not implement
that support: use the actual simulator map, and report an implementation
limitation if it cannot represent or schedule a requested layout. An existing
explicit map or concrete distribution must never be silently changed.

The exceptions apply in that order. Thus 2x3 is two M C M rows; 1x4 and
2x4 use M C C M rows. Normal meshes with at least 3 rows and 3 columns keep
the original E/V/C row rotation and scheduling instructions.

Existing files in `programas/<operacion>/` are generated outputs, not inputs
that prevent compilation. On every invocation, rebuild the IR, graphs, initial
memory and all PE instruction files from the current C source and
`src/cgra.json`. Overwrite existing PE files and remove obsolete generated
`PE*.txt` files as described in skill 08. Changing the mesh, PE types, vector
width or memory capacity requires recompilation and validation; do not stop
because the old program targets different hardware. `run_cgra.py` loads and
executes these outputs; it does not run this compilation pipeline automatically.

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
and west neighbors. In the standard border layout the west and east columns
are **memory PEs** (`M`) and the interior PEs compute. Small and explicit
layouts follow the compatibility contract. The specialized types are:

```text
M = memoria    the west and east columns; loads/stores scalars and vectors, never computes
E = escalar    scalar arithmetic only
V = vectorial  vector arithmetic only
C = computo    scalar and vector arithmetic (E + V)
G = generico   memory and scalar compute; vector ISA when configured
```

The `distribucion` of `src/cgra.json` (`ciclo_filas`) rotates the compute rows
escalar, vectorial, computo: row `f` is `E`, `V` or `C` for `f mod 3` = 0, 1, 2:

```text
memory PEs:   PE(f, 0) and PE(f, columnas - 1), for every row f
compute PEs:  PE(f, c) with 1 <= c <= columnas - 2

f mod 3 = 0:  M E E ... E M
f mod 3 = 1:  M V V ... V M
f mod 3 = 2:  M C C ... C M
```

This picture describes the existing border distribution, not the narrow-mesh
exceptions. A generic PE can fulfill both memory and compute roles locally.

## Instruction Set per PE Type

```text
                       memoria (M)  escalar (E)  vectorial (V)  computo (C)
MOV  reg, imm               x            x            x              x
LD   reg, mem[index]        x
ST   mem[index], reg        x
ADD  dst, src1, src2                     x                           x
SUB  dst, src1, src2                     x                           x
MUL  dst, src1, src2                     x                           x
DIV  dst, src1, src2                     x                           x
SEND dir, reg               x            x            x              x
RECV dir, reg               x            x            x              x
NOP                         x            x            x              x
```

G combines the memory and compute instruction permissions actually configured
for it. With no vector width it uses only the scalar ISA. Never infer G's
permissions just from the letter; read its declared or implicit generic type.

When `src/cgra.json` has `ancho_vector` (`W`) and vector compute PEs (`V` or
`C`), the pipeline runs in `vectorial` mode (skill 01): the vector compute PEs
do the work in blocks of `W` contiguous elements, and the scalar compute PEs
(`E`) do, at the same time, what does not fill a block, with the scalar
instructions above. The vector instructions are:

```text
                    memoria (M)  vectorial (V)  computo (C)
VLD  vreg, mem[index]       x                                W contiguous words
VST  mem[index], vreg       x
VADD/VSUB/VMUL/VDIV                      x             x     lane by lane
VREDSUM reg, vreg                        x             x     sum of the W lanes
VBCAST  vreg, reg                        x             x     a scalar to the W lanes
VSEND dir, vreg             x            x             x     the whole vector, 1 cycle
VRECV dir, vreg             x            x             x
```

Memory-capable PEs load operands and send them to the compute PEs when separate;
a `G` can load, compute and store locally (skills 05-07). `run_cgra.py` refuses
to load a program with an instruction outside the type of its PE.
