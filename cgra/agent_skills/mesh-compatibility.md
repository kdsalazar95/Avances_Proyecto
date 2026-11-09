# Mesh Compatibility Contract

This contract is shared by skills 01 and 05 to 09. It defines the valid mesh
sizes, the single-row layout and arbitrary explicit maps. The single-row
layout is implemented in `FormatoCGRA` (`malla_pequena()` in
`src/formato_cgra.py`) and enabled with `"adaptar_mallas_pequenas": true` in
`src/cgra.json`; its schedules are in skill 07. On every
run, use the map and instruction lists actually returned by `FormatoCGRA`.
Never compile for an imagined map that differs from the simulator's map.
Report an unsupported layout separately from an impossible hardware mapping.

## Valid Mesh Sizes

A target mesh has at least 1 row and **at least 3 columns** (up to the limits
of `src/cgra.json`). The west and east columns hold the memory PEs (M) and the
compute PEs go between them, so there is always at least one interior compute
column. One- and two-column meshes (1x1, 1x2, 2x1, Nx1, Nx2) are **not** valid
targets: report them as an unsupported configuration (skill 01) and do not
generate PE programs for them.

## Automatic Layout Policy

`FormatoCGRA` applies these cases when `src/cgra.json` has
`"adaptar_mallas_pequenas": true` and no explicit `malla`. Without that key,
the `distribucion` keeps its declared meaning at every size. An explicit
`malla` always takes precedence.

| Dimensions | Types and placement | Execution topology |
|---|---|---|
| 1 row, at least 3 columns | west/east `M`; every interior PE is `C` | two servers with 3 columns; nearest compatible border server and relays with more columns |
| at least 2 rows and at least 3 columns | west/east `M`; interior rows cycle `E`, `V` (row 0 `E`) | border layout of the `distribucion` |

Dimensions are always rows x columns. In particular, 1x3 is `M C M`, 1x4 is
`M C C M`, 2x3 is `M E M` / `M V M` and 2x4 is `M E E M` / `M V V M`. 1x10
keeps memory at both ends and C throughout the interior. Every valid mesh has
memory in both the west and the east column.

C already combines E and V arithmetic. G is not used by the automatic layouts.
No new ISA instructions are implied.

### When to Use C

The compute rows cycle `E`, `V`, so a mesh holds both scalar and vector compute
PEs as soon as it has **two compute rows**. Then C is never used: scalar work
goes to E, vector work to V, and the scalar tail of a vector split goes to E.
Only a mesh with a **single row** (1xN, N >= 3) cannot hold both E and V.
There every compute position is **C**, since C is E + V.

| Rows | Compute types | Meshes |
|---|---|---|
| 1 | only `C` | `1xN` (N >= 3) |
| 2 or more | `E`, `V`, `E`, ... (never `C`) | at least 2 rows and 3 columns |

`"adaptar_mallas_pequenas": true` makes `FormatoCGRA` apply this rule for any
valid `filas` and `columnas`, so changing the size never requires editing the
map: the agent compiles for the printed map and continues. If the key is
missing or false, the `distribucion` keeps its declared meaning at every size
(a 1x3 is `M E M`, scalar only): the compiler does not rewrite `cgra.json`,
compiles for the actual map (scalar on E, vector on V) and notes in the mesh
summary that the key enables the C layout of single-row meshes.

`G` is a generic PE, not a new instruction: it combines the scalar memory,
arithmetic, movement and communication instructions. It may also have the
vector ISA if `ancho_vector` exists. Without vector hardware, use scalar
instructions only. If an explicit typed map names `G`, that type must be
declared with its real instructions. Do not silently add memory instructions
to `C` or arithmetic instructions to `M`. A configuration without `tipos_pe`
already describes generic PEs; classify their capabilities in the same way.

## Capability Checks Before Mapping Work

Dimensions and memory capacity must be positive integers, with at least 3
columns (see Valid Mesh Sizes). Vector width, when
present, must also be a positive integer. Reject malformed maps, undefined PE
types or vector instruction declarations without vector hardware through the
shared configuration validator; do not silently repair them in the compiler.

Classify by allowed instructions, not solely by the letters or position:

- Memory-capable: the required loads and stores. Scalar access needs `LD/ST`;
  vector access needs `VLD/VST` and a configured vector width.
- Scalar-capable: the arithmetic required by this operation and its reduction.
  Reduction needs `ADD`; matmul/convolution need `MUL` and `ADD`.
- Vector-capable: the vector operations used by the selected schedule, including
  `VREDSUM` or `VBCAST` when the schedule requires them.
- A generic PE can belong to several sets simultaneously. Local computation
  does not need `SEND/RECV` to itself.

For each operation, establish all of the following before publishing a schedule:

1. At least one PE can access the input and output regions.
2. At least one PE can perform each required calculation.
3. Operands and results have compatible routes, or remain local on a `G`.
4. A complete reduction plan exists for all assigned partial sums.
5. Every scalar remainder has an eligible scalar PE and a route.
6. The initial memory fits `palabras_memoria`; tiling reuses PEs but does not
   provide external memory or reduce the existing `memoria.bin` requirement.

Prefer a valid scalar schedule if vector execution is unavailable but the
operation is scalar-executable. If no complete valid plan exists, report the
operation, missing capability or route, and affected PEs/regions. A valid but
currently unimplemented layout must instead be reported as an implementation
limitation. Do not request a larger mesh merely because execution takes tiles.

## Insufficient Resources and Remainders

| Situation | Required behavior |
|---|---|
| Few compute PEs | reuse them for multiple work units/tiles |
| Few memory PEs | share servers, serialize conflicting accesses and relay over valid paths |
| More PEs than work | leave unused PEs in `NOP`; initialize any partial accumulator that is actually reduced |
| No memory-capable PE | operation error: no way to load operands/store outputs |
| No required arithmetic | operation error, even on a large mesh |
| Scalar-only compute PEs | schedule all work scalarly; do not emit vector arithmetic |
| Vector-only compute PEs | full vectors only, with vector accumulation/reduction when needed |
| Tail or output row shorter than W | use an eligible `E/C/G`, or run the whole operation scalarly if possible |
| Tail with no reachable scalar compute PE | operation error; never read past a bank or borrow lanes from the next output row |
| No compatible vector route | find another vector PE/server/path, or fall back to scalar execution if feasible |
| Initial memory too large | error with required and available word counts |

For now do not introduce implicit zero padding, masked accesses, scalar-to-lane
insertion, or external streaming. Those need separate ISA/compiler contracts.
`VBCAST` broadcasts one scalar to every lane; it does not pack different scalars.

## Communication and Reductions

Build paths from north/south/east/west neighbors. Each scalar hop requires
`SEND` on its sender and `RECV` on its receiver; vector hops require
`VSEND/VRECV` at every endpoint and relay. Validate outbound and return paths.
Use the nearest compatible server by path length; break ties by row-major PE
order. A local memory-capable compute PE has path length zero. The existing
two-server `M C M` optimization remains available when both routes are valid.

For example, `M E V` has a scalar path but no direct vector path through `E`.
Do not claim that scalar relays alone can assemble a vector at the destination.
For an arbitrary explicit map, choose a feasible subset of compute PEs instead
of requiring every compute PE to participate. Unused PEs may still relay.

The border-layout reduction route in skill 06 is a specialization, not a
universal requirement. With an arbitrary map, select a compatible reduction tree or gather partials
at a reachable accumulator. A scalar tree needs `ADD` at its combining nodes;
a `V` can relay scalars but cannot add them. A connected vector tree can instead
combine vector partials with `VADD`, then run `VREDSUM` once at its root.
Never combine a scalar sum by broadcasting it and reducing all lanes: that
would multiply its contribution by W.

Choose a reachable output memory PE; prefer `PE00` when feasible, otherwise
the first feasible memory PE in row-major order. Record `output_pe`, the
combining root and routes in the manifest. With one contributing compute PE,
there is no inter-compute reduction, only a local sum and output transfer.

## Tiling Contract

Use the ordered eligible compute lists from the manifest.

- Scalar outputs: unit e goes to eligible PE `e mod P_s` in round `e // P_s`.
- Vector outputs: a unit is W contiguous columns in one output row; unit e
  goes to vector PE `e mod P_v` in round `e // P_v`.
- The rectangular scalar tile in skill 06 remains valid for the uniform
  border layout. Other layouts use explicit unit lists and round numbers.
- A `C/G` shared between vector and scalar phases cannot execute both at
  once. Schedule its remainder after its vector work, or reserve it for scalar
  work before computing capacities. Never count the same PE twice in parallel.
- If `P_v = 0`, use scalar work. If the scalar-only set is empty, use reachable
  `C/G` for scalar work; never divide by zero or assign scalar arithmetic to V.
- Costs in the adjacent-pair tables are estimates for that topology only.
  Include server sharing and relay cycles when comparing other assignments.
- Reset each output accumulator before its next output unit. In reduction,
  retain the local partial sum across input blocks until final combination.
- Edge rounds do not load/store nonexistent units. Pad idle PEs with `NOP`,
  except for explicitly scheduled relays, shared-server work or reduction work.

## Acceptance Cases for the Later Implementation

Validate the three source operations with automatic layouts `1x3`, `1x4`,
`2x3`, `2x4`, `1x10`, `2x10`, `3x4`, `4x3` and a normal wider mesh such as
`6x10`. Check that C appears only with a single row (1xN), that 2x3 is
`M E M` / `M V M`, that every mesh with two or more rows uses the E, V row
rotation, and that one- and two-column meshes are reported as unsupported. Cover both dimension orderings, exact and partial vector units,
multiple tile rounds, inactive edge units and coordinate widths beyond one digit.
Also cover explicit all-memory, all-compute, scalar-only, vector-only with/without
tails, mixed rows with blocked vector routes, consecutive V reduction nodes,
generic PEs, too-small memory and a valid route to output other than `PE00`.
Check computed results, exact output coverage, accumulator lifetimes, instruction
permissions, all transfers, memory bounds and complete regenerated PE files.
These are acceptance requirements, not a claim that those tests have passed.
