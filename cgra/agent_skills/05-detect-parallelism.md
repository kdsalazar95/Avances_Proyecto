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
   - `ADD`/`SUB`/`MUL`/`DIV` (scalar arithmetic) run only on **compute PEs**
     of type `escalar` (`E`) or `computo` (`C`), and the vector arithmetic on
     `vectorial` (`V`) or `computo` (`C`).

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
7. In `vectorial` mode (manifest, skill 01), group the work in **blocks of `W`
   contiguous elements** for the vector compute PEs, and keep the scalar work
   units above for the scalar compute PEs. Both run at the same time, each on
   its own rows, and the work is split between them by cost so that they end
   together (skill 06): the `escalar` rows do the elements that do not fill a
   block, and also whole blocks when that makes the run shorter. A `VLD`/`VST` only reads or writes
   contiguous words, so every block must be contiguous in memory:
   - **map loop:** a block is `W` consecutive iterations (`c[i:i+W]`).
   - **reduction loop:** each block is reduced inside the compute PE with
     `VREDSUM` and added to `acc`; the reduction across the compute PEs does
     not change.
   - **map with a nested reduction:** a block is `W` consecutive output
     elements of the **same row**, accumulated as a vector. In matmul,
     `result[i][j:j+W] += a[i][k] * b[k][j:j+W]`; in convolucion,
     `result[i][j:j+W] += kernel[ki][kj] * imagen[i+ki][j+kj : j+kj+W]`. The
     scalar factor is copied to every lane with `VBCAST`; the vector factor is
     a piece of a row, so it is contiguous. Do not vectorize along a column
     (`b[k:k+W][j]` is not contiguous).

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

In `vectorial` mode, for `reduccion`:

```text
parallel_work:
  block j (elements jW .. jW+W-1):
    memory PE:  VLD a, VLD b, send both              (VLD, VSEND)
    compute PE: receive both, add lane by lane       (VRECV, VADD)
    compute PE: send c back, reduce it into acc      (VSEND, VREDSUM, ADD)
    memory PE:  receive and store the block of c     (VRECV, VST)
tail_work:
  the elements after the last full block, as the scalar iterations above
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

In `vectorial` mode, for `matmul` and `convolucion`:

```text
parallel_work:
  row segment (i, j0 .. j0+W-1), owned by a compute PE:
    memory PE:  load the scalar factor and the vector factor of each step,
                send both                                 (LD, SEND, VLD, VSEND)
    compute PE: copy the scalar to every lane, multiply,
                accumulate in a vector register            (VBCAST, VMUL, VADD)
    compute PE: send the segment back; memory PE stores it (VSEND, VRECV, VST)
tail_work:
  the output columns after the last full segment of each row, as the scalar
  output elements above
```

## Notes

Tree or mesh reductions can change floating-point rounding compared with the
scalar C loop. Mention this when generating the schedule.

In `matmul` and `convolucion`, `clang` fuses each multiply-accumulate into one
`llvm.fmuladd` (see skill 04), which may also round differently from a separate
`MUL` and `ADD` in the CGRA.
