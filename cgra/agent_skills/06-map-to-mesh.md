# Skill: Map to Mesh

## Purpose

Assign the work units and the communication between PEs to a `filas x columnas`
mesh whose PEs have **types** (see [Receive Inputs](01-receive-inputs.md)). Apply
the [Mesh Compatibility Contract](mesh-compatibility.md) first. The geometric
formulas below specialize the standard border layout; they must not be used
for one/two-column or arbitrary explicit layouts. Always use the actual map.

## Local, Narrow and Explicit Layouts

Use `topologia`, eligible PE lists, servers, routes and `output_pe` from the
manifest. For local G, set its server to itself and use no transfer. For Nx1,
memory is at row 0, compute starts at row 1, and transfers go north/south.
For Nx2, each row's west M serves its east C. With one or two rows and at
least 3 columns, keep M at both ends and use only C in the interior. With
3 columns (1x3 or 2x3), use both servers for each row's C. Wider special
rows use the nearest border server and compatible relays through C, as in
the existing shared-server rules. For other explicit maps, find compatible paths as specified by the
contract; do not invent a border memory PE or arithmetic at a memory-only PE.

For at least 3 rows and 3 columns, retain the original normal-mesh mapping
and E/V/C row rotation. Special C-only compute rows use separate scalar and
vector phases or a reserved subset of C for scalar work; do not assume an E
row exists or schedule two instructions on a C in the same cycle.

Use eligible-list tiling for these cases, including a single worker. Record
every unit, owner, phase and round. With one worker, input/output transfers and
computation repeat until the full problem is covered. A singleton G needs
neither a compute-to-compute reduction nor an output transfer.

## Standard Border Mesh

```text
memory PEs:   PE(f, 0) and PE(f, columnas - 1)       for 0 <= f <= filas - 1
compute PEs:  PE(f, c) with 1 <= c <= columnas - 2   (the compute block)

M = memoria:   LD, ST, VLD, VST, SEND, RECV, VSEND, VRECV, MOV, NOP  (moves data, never computes)
E = escalar:   ADD, SUB, MUL, DIV, SEND, RECV, MOV, NOP              (scalar only, never touches memory)
V = vectorial: VADD..VDIV, VREDSUM, VBCAST, SEND, RECV, VSEND, VRECV, MOV, NOP  (vector only)
C = computo:   E + V                                                 (scalar and vector)
G = generico:  memory + scalar compute, and vector instructions when configured
```

`PE{fila}{columna}` is connected only to its north, south, east, and west
neighbors. Every PE, of either type, can `SEND`/`RECV` to any neighbor.

## PE Identifiers

Each coordinate is written with the same width, and that width is however many
digits the largest index needs:

```text
largest index 0..9       -> one digit per coordinate
largest index 10..99     -> two digits per coordinate
largest index 100..999   -> three digits per coordinate
largest index 1000..9999 -> four digits per coordinate
and so on, without a fixed digit limit.
```

Use the simulator's rule: `width = len(str(max(filas, columnas) - 1))`,
then `f"PE{fila:0{width}d}{columna:0{width}d}"`. For a `120x3023` mesh,
the width is 4 and `(119, 3022)` is `PE01193022`. Prefer `Malla.id_de()`
when generating identifiers in code so filenames agree with the simulator.

Never write `PE10` in a mesh of 11 or more, where it would be ambiguous. A
`10x10` mesh still uses one digit (`PE00` .. `PE99`).

## The Two Rules of the Types

1. **All arithmetic requires compute capability.** A memory-only M never
   emits ADD/SUB/MUL/DIV. A G can perform arithmetic as well as access memory.
2. **All memory accesses require memory capability.** E/V/C never emit `LD/ST`.
   A G may load, compute and store locally. With separate roles, operands and
   results use compatible send/receive routes.

## Server of Each Compute PE in the Border Layout

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

### Two Servers (`columnas = 3`)

With `3` columns every row is `memory, compute, memory`: the compute PE
`PE(f, 1)` is next to both memory PEs, so **both serve it** instead of leaving
the east one idle:

```text
west server  PE(f, 0):  loads and sends the first operand            (to C = east)
east server  PE(f, 2):  loads and sends the second operand,
                        receives the results and stores them         (to C = west)
```

The first operand is `a` in reduccion and matmul and the scalar factor in the
vector steps; the second is `b`, the image pixel or the vector factor. The
reduction route does not change: its last hop still goes to `PE(0, 0)`, which
stores `result[0]`. Skill 07 has the tables for this case.

## Reduccion: Scalar Border Assignment

Number the compute PEs in row-major order (`P_c = filas * (columnas - 2)` of
them). Iteration `i` goes to compute PE number `p = i mod P_c`:

```text
fila    = p // (columnas - 2)
columna = 1 + p mod (columnas - 2)
```

Use this interleaving rule for this specialization. Every selected PE must
support scalar arithmetic; V cannot own a scalar unit. Other layouts use
`eligible_compute_pes[i mod count]`. For iteration `i`,
the server of that compute PE loads `a[i]` and `b[i]`, the compute PE adds them
and accumulates the sum in `acc`, and the server stores `c[i]`. When `N` is
larger than `P_c`, each compute PE gets `ceil(N / P_c)` or `floor(N / P_c)`
iterations of that interleaved sequence.

## Reduccion: Border Reduction Route

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

The scalar border route has `(columnas - 3) + (filas - 1) + 1` hops in
sequence. Vector-only rows need the adjustments below. Other layouts use the
recorded reduction plan and output PE; PE00 is preferred, not mandatory.

## Matmul and Convolucion: Scalar Border Output Tiles

This rectangular formula applies when all listed compute positions are scalar
eligible. With V rows, use the vector/scalar work lists below. For narrow or
irregular maps, use the ordered eligible lists from the compatibility contract.

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

## Vectorial Mode

In `vectorial` mode (manifest, skill 01) the servers and the reduction route do
not change; only the work units do. `W` is `ancho_vector`. The memory PEs (`M`)
serve both scalars and vectors, so every compute PE is served by the `M` of its
row as before. A vector compute PE (`V` or `C`) does the blocks of `W` elements
(a C/G may also use scalar instructions, e.g. to accumulate VREDSUM results)
and a scalar-assigned PE handles scalar units. The border examples use E for
that group; the compatibility contract covers C/G phases. Number each assigned
class apart, in row-major
order:

```text
vector compute PEs:  q = 0 .. P_v - 1      (vector_compute_pes)
scalar compute PEs:  r = 0 .. P_s - 1      (scalar_compute_pes)
```

In the standard row-homogeneous layout, disjoint vector and scalar groups may
work **at the same time** on separate rows. Other maps need explicit link/server
reservations, and shared C/G need separate phases. Split the work so both groups
end together: a vector PE is faster, but leaving the scalar rows idle wastes
them. The cost of one unit comes from the tables of skill 07:

```text
                         vector unit (W elements)    scalar element
reduccion                6                           6  (5 with two servers)
matmul / convolucion     5 * steps + 4               4 * steps + 4
```

`steps` is the length of the accumulation (`N` in matmul, `K * K` in
convolucion). Give the vector PEs the first `n` units, and the scalar PEs
every element they do not cover, with `n` chosen as:

```text
T_v(n) = ceil(n / P_v) * cost_vector
T_s(n) = ceil((elements - n * W) / P_s) * cost_scalar
n      = the value from 0 to the number of whole units that gives the lowest
         max(T_v(n), T_s(n)); on a tie, the largest n
```

Try every `n` for this adjacent, disjoint-resource specialization. If P_v is
zero, run scalar work. If the scalar-only set is empty, use reachable C/G for
the tail in a later phase or reserve them before splitting. A V cannot do the
scalar tail. If no such route/capability exists, report the unsupported tail.
Do not evaluate these formulas with a zero denominator. For shared/distant
servers, include actual delivery and return costs; these table costs alone do
not establish an optimal partition.

### Reduccion

```text
blocks:  the first n blocks (n from the split above, at most N // W);
         block j covers a[jW .. jW+W-1], b[...], c[...]
         block j -> vector compute PE q = j mod P_v
scalar:  the elements i = n*W .. N-1;  the t-th of them (t = i - n*W)
         -> scalar compute PE r = t mod P_s, as a scalar iteration
```

Then every compute PE joins the reduction route above with its partial sum.
`E` and `C` keep it in `acc`. A `V` has no scalar `ADD`, so it keeps it as a
vector in `v4` and takes part in the route like this:

```text
row hops in a V row:   VSEND west, v4 / VRECV east, v5, then VADD v4, v4, v5;
                       after the row hops, PE(f, 1) emits VREDSUM acc, v4
column hop into a V:   the V does not add: it receives into rT and, on its own
                       hop north, sends rT first and then acc; the E or C above
                       adds both (one hop each)
```

This particular scalar-forwarding route ends in E/C and assumes no consecutive
V rows, as in the standard rotation E,V,C. Arbitrary explicit maps must use
the contract's scalar or vector reduction plan, including consecutive V nodes,
rather than applying this specialization without checking capabilities.

### Matmul and Convolucion

The output is split by rows into **segments** of `W` contiguous columns:

```text
segs per row = salida_columnas // W
segment      = (i, j0) with j0 = s * W, s = 0 .. segs - 1, numbered e in
               row-major order (e = i * segs + s)
vector work  = the first n segments (n from the split above)
tile t       -> vector compute PE q owns segment e = t * P_v + q
               (none if e is past the n-th segment: that pair emits NOP)
```

In every step of the accumulation the server loads one scalar and one vector
for its compute PE, and sends both:

```text
matmul       (N = TAMANO_MATRIZ)
  rA <- a[i * N + k]                    (LD,  scalar factor)
  v1 <- b[k * N + j0]                   (VLD, b[k][j0 .. j0+W-1])

convolucion  (I = image side, K = kernel side, S = I - K + 1)
  rA <- kernel[ki * K + kj]             (LD,  scalar factor)
  v1 <- imagen[(i + ki) * I + j0 + kj]  (VLD, one piece of an image row)
```

After the last step, the compute PE sends the segment back and the server
stores it:

```text
VST result[i * N + j0], v3          (matmul)
VST result[i * S + j0], v3          (convolucion)
```

The **scalar work** is every output element that is not in one of those `n`
segments: the segments left to the scalar rows, and the columns
`j = segs * W .. salida_columnas - 1` of every row. Number those elements in
row-major order; the `e`-th goes to scalar compute PE `e mod P_s` in tile
`e // P_s`, and is computed with the scalar operands and stores above
(`LD rB` instead of `VLD`). The vector tiles and the scalar tiles run at the
same time, on their own rows.

```text
vector tiles = ceil(n / P_v)                                        (vector PEs)
scalar tiles = ceil((salida_filas * salida_columnas - n * W) / P_s)    (scalar PEs)
```

## Sizes That Do Not Match the Mesh

- **Output larger than the compute block:** use multiple rounds. Only the
  rectangular scalar border tile uses
  `ceil(salida_filas / tile_filas) * ceil(salida_columnas / tile_columnas)`;
  vector and eligible-list modes use their actual work-unit/worker counts.
- **Edge tiles:** when the output is not a multiple of the block, the compute
  PEs whose `global_fila` or `global_columna` falls outside the output have
  nothing to calculate in that tile and emit NOP except when explicitly
  relaying. Their shared server may serve another active PE; do not idle it
  merely because one client has no work. Every output is assigned once.
- **Mesh larger than the problem:** the extra compute PEs stay in `NOP`. In a
  reduction, initialize a neutral scalar/vector partial if the PE joins the
  reduction route. Otherwise leave it idle or use it only as a relay.

## Outputs

```text
server of every compute PE      (memory PE and direction; west and east with 3 columns)
work list per compute PE        (iterations, or output elements per tile)
memory list per memory PE       (LD/ST it performs for its compute PEs)
tile list                       (for matmul and convolucion; vector tiles,
                                 and scalar tiles in vectorial mode, with the split n)
communication edges             (operand delivery, results, reduction route)
```

## Notes

Only north, south, east, and west communication exists. A diagonal transfer must
be implemented as several neighbor hops.
