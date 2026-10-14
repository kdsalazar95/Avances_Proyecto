# Skill: Map to Mesh

## Purpose

Assign the work units and the communication between PEs to a `filas x columnas`
mesh whose PEs have **types** (see [Receive Inputs](01-receive-inputs.md)). The
west and east columns are always **memory PEs** and the rest are **compute
PEs**; only the size changes, and it comes from the manifest (`src/cgra.json`).
Every rule below is written for any `filas` and `columnas`; apply it to the map
of the manifest (`malla`).

## Mesh

```text
memory PEs:   PE(f, 0) and PE(f, columnas - 1)       for 0 <= f <= filas - 1
compute PEs:  PE(f, c) with 1 <= c <= columnas - 2   (the compute block)

M = memoria: LD, ST, SEND, RECV, MOV, NOP               (moves data, never computes)
C = computo: ADD, SUB, MUL, DIV, SEND, RECV, MOV, NOP   (never touches memory)
```

`PE{fila}{columna}` is connected only to its north, south, east, and west
neighbors. Every PE, of either type, can `SEND`/`RECV` to any neighbor.

## PE Identifiers

Each coordinate is written with the same width, and that width is however many
digits the largest index needs:

```text
up to 10 rows and columns (indices 0..9) -> one digit each:  PE00, PE12, PE99
11 or more rows or columns (index 10+)    -> two digits each: PE0100 is row 1, column 0
                                                              PE1000 is row 10, column 0
```

Never write `PE10` in a mesh of 11 or more, where it would be ambiguous. A
`10x10` mesh still uses one digit (`PE00` .. `PE99`).

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
for the compute PE PE(f, c):
  server     = PE(f, 0)             if c <= columnas - 1 - c   (west, also on a tie)
               PE(f, columnas - 1)  otherwise                  (east)
  distance d = c                    for a west server
               columnas - 1 - c     for an east server
  direction to the compute PE  = east for a west server, west for an east server
  direction back to the server = the opposite one
```

Only `PE(f, 1)` and `PE(f, columnas - 2)` are next to their server (`d = 1`).
With exactly two compute columns every memory PE serves one compute PE; with
more, each memory PE serves several, and the farther ones are reached through
the compute PEs in between.

A value for a compute PE at distance `d > 1` crosses the compute PEs in between
one hop per cycle; they relay it through `rT` (`RECV` then `SEND` on the next
cycle). If a memory PE serves several compute PEs, it serves them one after
another (skill 07).

## Reduccion: Assignment

Number the compute PEs in row-major order (`P_c = filas * (columnas - 2)` of
them). Iteration `i` goes to compute PE number `p = i mod P_c`:

```text
fila    = p // (columnas - 2)
columna = 1 + p mod (columnas - 2)
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

In coordinates:

```text
row hops:     PE(f, c) -> PE(f, c - 1)   for c = columnas - 2 down to 2, in every row f
column hops:  PE(f, 1) -> PE(f - 1, 1)   for f = filas - 1 down to 1
store:        PE(0, 1) -> PE(0, 0), and PE(0, 0) stores result[0]
```

The route has `(columnas - 3) + (filas - 1) + 1` hops in sequence, whatever the
size.

## Matmul and Convolucion: Output Tiles

One compute PE per output element, with the **compute block** sliding over the
output in tiles of `tile_filas x tile_columnas` (the compute block of the
manifest: `filas x (columnas - 2)`). Number the compute PEs inside the block by
`(bf, bc)`, from 0. For every tile starting at
`(inicio_fila, inicio_columna)`, the compute PE at block position `(bf, bc)`
owns:

```text
global_fila    = inicio_fila + bf
global_columna = inicio_columna + bc
```

The compute PE at block position `(bf, bc)` is `PE(bf, bc + 1)`.

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
sides of every row, each memory PE loading it for its own compute PE is cheaper than relaying
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
