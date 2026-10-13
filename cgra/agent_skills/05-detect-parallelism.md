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
2. Distribute the work over all `filas * columnas` PEs of the mesh, taken from
   the manifest. If the work does not fit in the mesh, do not ask for more PEs:
   hand it to skill 06, which splits it into tiles of the mesh size.
3. For a **reduction loop**, use local partial sums per PE followed by a single
   mesh reduction that ends in `PE00`. This is the only case that needs
   `SEND`/`RECV`.
4. For a **map with a nested reduction**, every PE owns one output element at a
   time. An operand shared by a whole row, column or mesh enters once and
   travels from neighbor to neighbor, so a PE reads memory only when it is the
   one bringing the value in, or when the value is different for every PE.
5. If bit-identical sequential floating-point behaviour is required, serialize
   the reduction in original index order instead of reducing across the mesh.

## Outputs

Return work units. For `reduccion`:

```text
parallel_work:
  iteration i:
    load a[i], load b[i]
    add
    store c[i]

reduction_work:
  local partial sums per PE
  final mesh reduction to PE00
```

For `matmul` and `convolucion`:

```text
parallel_work:
  output element (i, j):
    receive or load the operands of the accumulation
    multiply and accumulate locally
    store the element

operand_traffic:
  values shared by a row, a column or the whole mesh cross it hop by hop
  accumulation stays local to each PE
```

## Notes

Tree or mesh reductions can change floating-point rounding compared with the
scalar C loop. Mention this when generating the schedule.

In `matmul` and `convolucion`, `clang` fuses each multiply-accumulate into one
`llvm.fmuladd` (see skill 04), which may also round differently from a separate
`MUL` and `ADD` in the CGRA.
