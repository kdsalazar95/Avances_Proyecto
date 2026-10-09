# Skill: Map to Mesh

## Purpose

Assign the work units and the communication between PEs to a `filas x columnas`
mesh whose PEs have **types** (see [Receive Inputs](01-receive-inputs.md)). The
size and the type map come from `src/cgra.json`; the `4x4` below is the default
file, used as the example.

## Mesh

```text
        col 0      col 1      col 2      col 3
row 0   PE00 (M) - PE01 (C) - PE02 (C) - PE03 (M)
         |          |          |          |
row 1   PE10 (M) - PE11 (C) - PE12 (C) - PE13 (M)
         |          |          |          |
row 2   PE20 (M) - PE21 (C) - PE22 (C) - PE23 (M)
         |          |          |          |
row 3   PE30 (M) - PE31 (C) - PE32 (C) - PE33 (M)

M = memoria: LD, ST, SEND, RECV, MOV, NOP      (moves data, never computes)
C = computo: ADD, SUB, MUL, DIV, SEND, RECV, MOV, NOP   (never touches memory)
```

`PE{fila}{columna}` is connected only to its north, south, east, and west
neighbors. Every PE, of either type, can `SEND`/`RECV` to any neighbor.

## PE Identifiers

Each coordinate is written with the same width, and that width is however many
digits the largest index needs:

```text
fewer than 10 rows and columns -> one digit each:  PE00, PE12, PE77
10 or more rows or columns     -> two digits each: PE0100 is row 1, column 0
                                                   PE1000 is row 10, column 0
```

Never write `PE10` in a mesh of 10 or more, where it would be ambiguous.

## The Two Rules of the Types

1. **All arithmetic runs on compute PEs.** A memory PE never emits `ADD`,
   `SUB`, `MUL` or `DIV`.
2. **All memory accesses run on memory PEs.** A compute PE never emits `LD` or
   `ST`. Every value a compute PE uses arrives with `RECV`, and every value it
   produces for memory leaves with `SEND`.

## Server of Each Compute PE

Each compute PE is served by the **nearest memory PE in its own row** (on a tie,
the west one). The server loads the operands of that compute PE, sends them to
it, receives its results and stores them.

```text
direction to the compute PE  = east if the server is to the west, else west
direction back to the server = the opposite one
```

With the default file every compute PE is right next to its server, and every
memory PE serves exactly one compute PE:

```text
PE(f,1) <- server PE(f,0), operands go east, results come back west
PE(f,2) <- server PE(f,3), operands go west, results come back east
```

If a CGRA has a compute PE at distance `d > 1` from its server, the value
crosses the compute PEs in between one hop per cycle; they relay it through
`rT` (`RECV` then `SEND` on the next cycle). If a memory PE serves several
compute PEs, it serves them one after another. A row with no memory PE cannot
be used: stop and tell the user (skill 01).

## Reduccion: Assignment

Number the compute PEs in row-major order (`P_c` of them). Iteration `i` goes
to compute PE number `i mod P_c`:

```text
4x4 default: 0 -> PE01, 1 -> PE02, 2 -> PE11, 3 -> PE12,
             4 -> PE21, 5 -> PE22, 6 -> PE31, 7 -> PE32, 8 -> PE01, ...
```

Use this single rule; do not switch to contiguous chunks. For iteration `i`,
the server of that compute PE loads `a[i]` and `b[i]`, the compute PE adds them
and accumulates the sum in `acc`, and the server stores `c[i]`. When `N` is
larger than `P_c`, each compute PE gets `ceil(N / P_c)` or `floor(N / P_c)`
iterations of that interleaved sequence.

## Reduccion: Reduction Route

Only compute PEs hold partial sums, so the route runs over the compute block
and ends in a memory PE:

```text
1. Row reduction: in every row, reduce the compute columns toward the westmost
   compute column (SEND west / RECV east hops, all rows at the same time).
2. Column reduction: reduce that column up to row 0
   (SEND north / RECV south hops).
3. Store: the compute PE at row 0 sends the total to its server, which stores
   result[0].
```

With the default file this is `PE02 -> PE01`, `PE12 -> PE11`, `PE22 -> PE21`,
`PE32 -> PE31`; then `PE31 -> PE21 -> PE11 -> PE01`; then `PE01 -> PE00`, and
`PE00` stores `result[0]`. For a larger mesh, add one hop per extra compute
column and per extra row, in the same directions.

## Matmul and Convolucion: Output Tiles

One compute PE per output element, with the **compute block** sliding over the
output in tiles of `tile_filas x tile_columnas` (the compute block of the
manifest; `4 x 2` with the default file). Number the compute PEs inside the
block by `(bf, bc)`, from 0. For every tile starting at
`(inicio_fila, inicio_columna)`, the compute PE at block position `(bf, bc)`
owns:

```text
global_fila    = inicio_fila + bf
global_columna = inicio_columna + bc
```

With the default file `bc = 0` is column 1 and `bc = 1` is column 2. A `9x9`
output needs `ceil(9/4) * ceil(9/2) = 3 * 5 = 15` tiles.

### Operands of Each Step

The accumulation is split into steps (`k` in matmul, every `(ki, kj)` of the
window in convolucion). In every step, the server loads the two operands of
**its** compute PE and sends them:

```text
matmul       (N = TAMANO_MATRIZ)
  rA <- a[global_fila * N + k]
  rB <- b[k * N + global_columna]

convolucion  (I = image side, K = kernel side, S = I - K + 1)
  rA <- imagen[(global_fila + ki) * I + global_columna + kj]
  rB <- kernel[ki * K + kj]
```

An operand shared by a row (`a` in matmul) or by the whole tile (`kernel`) is
loaded by every memory PE that needs it, once per memory PE: with memory on both
sides, each memory PE loading it for its own compute PE is cheaper than relaying
it through compute PEs that are busy computing. When a memory PE serves several
compute PEs, it loads a shared operand once and sends it to all of them.

### Stores

After the last step of the tile, every compute PE sends `acc` to its server,
which stores it:

```text
ST result[global_fila * N + global_columna], rC          (matmul)
ST result[global_fila * S + global_columna], rC          (convolucion)
```

Walk the output tile by tile, row of tiles first, and restart the accumulators
on every tile.

## Sizes That Do Not Match the Mesh

- **Output larger than the compute block:** that is the tiling above. The number
  of tiles is `ceil(salida_filas / tile_filas) * ceil(salida_columnas / tile_columnas)`.
- **Edge tiles:** when the output is not a multiple of the block, the compute
  PEs whose `global_fila` or `global_columna` falls outside the output have
  nothing to do in that tile and emit `NOP`, and so does their server. Every
  output element must still be assigned exactly once.
- **Mesh larger than the problem:** the extra compute PEs stay in `NOP`. In a
  reduction they still execute `MOV acc, 0.0` at the start, so they contribute
  a neutral value and the reduction route above stays the same.

## Outputs

```text
server of every compute PE      (memory PE and direction)
work list per compute PE        (iterations, or output elements per tile)
memory list per memory PE       (LD/ST it performs for its compute PEs)
tile list                       (for matmul and convolucion)
communication edges             (operand delivery, results, reduction route)
```

## Notes

Only north, south, east, and west communication exists. A diagonal transfer must
be implemented as several neighbor hops.
