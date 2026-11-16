# Mesh Compatibility Contract

This contract is shared by skills 01 and 05 to 09. It defines the support for
small meshes and arbitrary explicit maps. On every run, use the map and
instruction lists actually returned by `FormatoCGRA`. Never compile for an
imagined map that differs from the simulator's map. Report an invalid or
unsupported layout separately from an impossible hardware mapping.

The only PE types are `M` (memoria), `E` (escalar) and `V` (vectorial). No PE
both accesses memory and computes, and no PE does both scalar and vector
arithmetic. Do not invent other types or silently add memory instructions to
`E`/`V` or arithmetic instructions to `M`.

## Automatic Layout Policy

With `adaptar_mallas_pequenas` in `src/cgra.json` (and no explicit `malla`),
`formato_cgra.malla_pequena()` rejects single-row meshes; otherwise the
`distribucion` applies as declared. Every mesh, with or without that flag, needs
at least 3 columns.

| Dimensions | Types and placement | Execution topology |
|---|---|---|
| `1xN` (any single row) | not valid: `FormatoCGRA` raises an error | none |
| fewer than 3 columns, e.g. `2x1`, `2x2`, `3x2` | not valid: `FormatoCGRA` raises an error | none |
| at least 2 rows and at least 3 columns | west/east `M`; interior rows rotate `E`, `V` | border layout from `distribucion` |

Dimensions are always rows x columns. A mesh needs at least two rows so it can
hold both an `E` and a `V`, and at least three columns (`M`, compute, `M`).
Report invalid sizes as an invalid configuration and do not generate PE files
for them. Any map, automatic or explicit, that does not have at least one `E`
and one `V` is also invalid, because the operations need both scalar and vector
compute. Transposing a shape can change its assignment: 3x4 and 4x3 have
different memory servers and routes.

## Capability Checks Before Mapping Work

Dimensions and memory capacity must be positive integers. Vector width, when
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
- The memory set (`M`) and the compute sets (`E`, `V`) never overlap, so every
  operand and result travels with `SEND/RECV` or `VSEND/VRECV`.

For each operation, establish all of the following before publishing a schedule:

1. At least one PE can access the input and output regions.
2. At least one PE can perform each required calculation.
3. Operands and results have compatible routes.
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
| Tail or output row shorter than W | use an eligible `E`, or run the whole operation scalarly if possible |
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
order. The two-server `M E M` / `M V M` optimization remains available when
both routes are valid.

For example, `M E V` has a scalar path but no direct vector path through `E`.
Do not claim that scalar relays alone can assemble a vector at the destination.
For an arbitrary explicit map, choose a feasible subset of compute PEs instead
of requiring every compute PE to participate. Unused PEs may still relay.

The border-layout reduction route in skill 06 is a specialization, not a
universal requirement. In a column, reduce upward through compatible compute
PEs and send the result to the memory at row 0.
With an arbitrary map, select a compatible reduction tree or gather partials
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

Use the ordered eligible compute lists from the manifest. Do not calculate a
zero-width tile using `columnas - 2` for one- or two-column meshes.

- Scalar outputs: unit e goes to eligible PE `e mod P_s` in round `e // P_s`.
- Vector outputs: a unit is W contiguous columns in one output row; unit e
  goes to vector PE `e mod P_v` in round `e // P_v`.
- The rectangular scalar tile in skill 06 remains valid for the uniform
  border layout. Other layouts use explicit unit lists and round numbers.
- Scalar work goes only to `E` and vector work only to `V`. Never count the
  same PE twice in parallel.
- If `P_v = 0`, use scalar work on `E`. If there is no reachable `E` for the
  scalar work, it is an operation error; never divide by zero or assign scalar
  arithmetic to V.
- Costs in the adjacent-pair tables are estimates for that topology only.
  Include server sharing and relay cycles when comparing other assignments.
- Reset each output accumulator before its next output unit. In reduction,
  retain the local partial sum across input blocks until final combination.
- Edge rounds do not load/store nonexistent units. Pad idle PEs with `NOP`,
  except for explicitly scheduled relays, shared-server work or reduction work.

## Acceptance Cases for the Later Implementation

Validate the three source operations with automatic layouts `2x3`, `3x3`,
`2x4`, `2x10`, `3x4`, `4x3` and a wider mesh such as `6x10`, and check that
`1x1`, `1x3`, `1x10`, `2x1`, `2x2`, `3x1` and `3x2` are rejected as invalid.
Check that 2x3 uses `M E M` / `M V M` and normal meshes keep the E/V row
rotation. Cover both dimension orderings, exact and partial vector units,
multiple tile rounds, inactive edge units and coordinate widths beyond one digit.
Also cover explicit all-memory, all-compute, scalar-only, vector-only with/without
tails, mixed rows with blocked vector routes, consecutive V reduction nodes,
too-small memory and a valid route to output other than `PE00`.
Check computed results, exact output coverage, accumulator lifetimes, instruction
permissions, all transfers, memory bounds and complete regenerated PE files.
These are acceptance requirements, not a claim that those tests have passed.
