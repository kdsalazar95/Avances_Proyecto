# Skill: Detect Parallelism

## Purpose

Determine which operations can execute independently across the PEs of the mesh
and which must be reduced or serialized.

## Actions

1. Classify every computation loop nest as one of:
   - **map loop:** independent elementwise iterations, such as
     `c[i] = a[i] + b[i]` in `reduccion`.
   - **reduction loop:** loop-carried accumulation over the whole problem, such
     as `result += c[i]` in `reduccion`.
   - **map with a nested reduction:** independent output elements, each with its
     own local accumulation. This is `matmul` (`C[i][j]` accumulates over `k`)
     and `convolucion` (`result[i][j]` accumulates over the kernel window).
2. Split every operation by the PE type that can run it (types from
   `src/cgra.json`, see skill 01):
   - `LD`/`ST` (memory accesses) run only on **memory PEs**;
   - `ADD`/`SUB`/`MUL`/`DIV` (arithmetic) run only on **compute PEs**.

   So every operand goes memory PE -> `SEND`/`RECV` -> compute PE, and every
   value to store goes back compute PE -> `SEND`/`RECV` -> memory PE.
3. Distribute the arithmetic over the compute PEs of the manifest. If the work
   does not fit, do not ask for more PEs: hand it to skill 06, which splits it
   into tiles of the compute block.
4. For a **reduction loop**, use local partial sums per compute PE followed by
   a single reduction across the compute PEs. The final sum is sent to a memory
   PE, which stores it.
5. For a **map with a nested reduction**, every compute PE owns one output
   element at a time and accumulates it in `acc`. Its memory PE loads the
   operands and delivers them; at the end the compute PE sends `acc` back and
   the memory PE stores it.
6. If bit-identical sequential floating-point behaviour is required, serialize
   the reduction in original index order instead of reducing across the mesh.

## Outputs

Return work units. For `reduccion`:

```text
parallel_work:
  iteration i:
    memory PE:  load a[i], load b[i], send both     (LD, SEND)
    compute PE: receive both, add                   (RECV, ADD)
    compute PE: send c[i] back                      (SEND)
    memory PE:  receive and store c[i]              (RECV, ST)

reduction_work:
  local partial sums per compute PE
  reduction across the compute PEs, then sent to a memory PE that stores it
```

For `matmul` and `convolucion`:

```text
parallel_work:
  output element (i, j), owned by a compute PE:
    memory PE:  load the two operands of each step and send them
    compute PE: receive them, multiply and accumulate locally
    compute PE: send the element back; memory PE stores it

operand_traffic:
  memory PE <-> compute PE, along the row (see skill 06)
  accumulation stays local to each compute PE
```

## Notes

Tree or mesh reductions can change floating-point rounding compared with the
scalar C loop. Mention this when generating the schedule.

In `matmul` and `convolucion`, `clang` fuses each multiply-accumulate into one
`llvm.fmuladd` (see skill 04), which may also round differently from a separate
`MUL` and `ADD` in the CGRA.
