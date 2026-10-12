# Skill: Map to Mesh

## Purpose

Assign the work units and any reduction communication to the PEs of a
`filas x columnas` mesh. The size comes from the manifest; `2x2` below is only
an example.

## Mesh

```text
PE00 --east/west-- PE01
 |                 |
north/south        north/south
 |                 |
PE10 --east/west-- PE11
```

The same layout extends to any size: `PE{fila}{columna}` sits at row `fila`,
column `columna`, connected only to its north, south, east, and west neighbors.

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

## Reduccion: Assignment

For `N` iterations and `P = filas * columnas` PEs, iteration `i` goes to:

```text
pe_index = i mod P
fila     = pe_index // columnas
columna  = pe_index mod columnas
```

With `filas=2, columnas=2` this is `0 -> PE00, 1 -> PE01, 2 -> PE10,
3 -> PE11`, so PE00 handles `a[0], a[4], a[8], ...`.

Use this single rule; do not switch to contiguous chunks. When `N` is larger
than `P`, each PE simply gets `ceil(N / P)` iterations of that interleaved
sequence, accumulates them locally, and only then runs the reduction route once.
That is the tiling strategy for a reduction: the tile is the local chunk each PE
reduces before the single mesh-wide reduction.

## Reduccion: Reduction Route

Reduce in two generalized passes:

```text
1. Row reduction: in every row, reduce the columns toward column 0
   (successive SEND west / RECV east hops, all rows at the same time).
2. Column reduction: reduce column 0 up to PE00
   (successive SEND north / RECV south hops).
```

With `filas=2, columnas=2` this is exactly `PE01 -> PE00`, `PE11 -> PE10`, then
`PE10 -> PE00`. For a larger mesh, add one hop per extra column and per extra
row, in the same directions; do not invent new communication patterns. The final
result lives in `PE00`, which stores it.

## Matmul and Convolucion: Output Tiles

One PE per output element, with the mesh sliding over the output in tiles of
`tile_filas x tile_columnas` (the mesh size by default). For every tile starting
at `(inicio_fila, inicio_columna)`, the PE at `(fila, columna)` owns:

```text
global_fila    = inicio_fila + fila
global_columna = inicio_columna + columna
```

An operand that several PEs share is not read from memory by each of them: it
enters the mesh once and then travels from neighbor to neighbor, one hop per
cycle, with `SEND`/`RECV`. Only the PE that brings a value in touches memory.

### Matmul

With `N = TAMANO_MATRIZ`, one step of the accumulation over `k` moves two
values, each along the direction where it is shared:

```text
a[global_fila * N + k]    is the same for the whole row    -> enters at column 0,
                                                              travels east
b[k * N + global_columna] is the same for the whole column -> enters at row 0,
                                                              travels south
```

Per `k`:

1. `PE(fila, 0)` loads `a[global_fila * N + k]` into `rA`.
2. That value crosses the row: `PE(fila, c)` emits `SEND east, rA` while
   `PE(fila, c + 1)` emits `RECV west, rA`, for `c = 0 .. columnas - 2`.
3. `PE(0, columna)` loads `b[k * N + global_columna]` into `rB`.
4. That value crosses the column: `SEND south, rB` / `RECV north, rB`, for
   `f = 0 .. filas - 2`.
5. Every PE of the tile computes `MUL rC, rA, rB` and `ADD acc, acc, rC`.

### Convolucion

With image side `I`, kernel side `K` and output side `S = I - K + 1`, the
kernel coefficient is shared by the whole mesh, while each PE needs its own
pixel:

```text
kernel[ki * K + kj]                              -> enters at PE(0,0), travels
                                                    east along row 0 and then
                                                    south down every column
imagen[(global_fila + ki) * I + global_columna + kj]
                                                 -> each PE loads its own pixel
```

Per `(ki, kj)` step: the coefficient travels, every PE loads its pixel with
`LD rA`, and both are combined with `MUL` and `ADD`.

### Stores

After the last step of the tile, every PE stores its element:

```text
ST result[global_fila * N + global_columna], acc          (matmul)
ST result[global_fila * S + global_columna], acc          (convolucion)
```

Walk the output tile by tile, row of tiles first, and restart the accumulators
on every tile.

## Sizes That Do Not Match the Mesh

- **Output larger than the mesh:** that is the tiling above. The number of tiles
  is `ceil(salida_filas / filas) * ceil(salida_columnas / columnas)`.
- **Edge tiles:** when the output is not a multiple of the mesh, the PEs whose
  `global_fila` or `global_columna` falls outside the output have nothing to do
  in that tile and emit `NOP`. Every output element must still be assigned
  exactly once.
- **Mesh larger than the problem:** the extra PEs stay in `NOP`. In a reduction
  they still execute `MOV acc, 0.0` at the start, so they contribute a neutral
  value and the reduction route above stays the same.

## Outputs

```text
PE{fila}{columna} work list   (one per PE in the mesh)
tile list                     (for matmul and convolucion)
communication edges           (reduction route, or travelling operands)
```

## Notes

Only north, south, east, and west communication exists. A diagonal transfer must
be implemented as several neighbor hops.
