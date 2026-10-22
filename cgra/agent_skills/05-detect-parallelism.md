# Skill: Detect Parallelism

## Purpose

Determine which operations can execute independently across the PEs of the mesh
and which must be reduced or serialized.

## Actions

Apply the [Mesh Compatibility Contract](mesh-compatibility.md). Determine
feasibility per operation and per work unit before distributing work. A large
mesh can still lack the required memory, arithmetic or routes.

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
   - `LD`/`ST` (memory accesses) run only on memory-capable PEs (`M/G`);
   - `ADD`/`SUB`/`MUL`/`DIV` (scalar arithmetic) run only on **compute PEs**
     of type `escalar` (`E`), `computo` (`C`) or generic (`G`), and vector
     arithmetic on `V/C/G` when their instruction lists allow it.

   Separate memory and compute PEs use compatible transfers in both directions.
   On a G that serves itself, load, compute and store locally in separate
   cycles, with no self-transfer.
3. Distribute the arithmetic over the compute PEs of the manifest. If the work
   does not fit, do not ask for more PEs: hand it to skill 06, which splits it
   into tile rounds over the eligible compute list. One compute PE is enough
   for sequential scalar work if its memory routes and arithmetic are valid.
4. For a **reduction loop**, use local partial sums per compute PE followed by
   a reduction over the contributing PEs with compatible combining nodes. A V
   keeps vector partials until `VREDSUM`; do not assign it scalar `ADD`.
   With one contributor no inter-compute reduction is necessary. Store the
   final sum at the selected memory-capable output PE, locally on G if possible.
5. For a **map with a nested reduction**, every compute PE owns one output
   element or vector segment at a time and accumulates it in acc or v3.
   A separate server delivers operands and stores the returned result;
   a memory-capable G may perform those accesses locally.
6. If bit-identical sequential floating-point behaviour is required, serialize
   the reduction in original index order instead of reducing across the mesh.
7. In `vectorial` mode (manifest, skill 01), group the work in **blocks of `W`
   contiguous elements** for the vector compute PEs, and keep the scalar work
   units above for the scalar compute PEs. They may run concurrently on disjoint
   PEs and routes; a shared C/G executes separate phases. Split work by scheduled
   cost (skill 06): eligible scalar PEs do the elements that do not fill a
   block, and also whole blocks when that makes the run shorter. A `VLD`/`VST` only reads or writes
   contiguous words, so every block must be contiguous in memory:
   - **map loop:** a block is `W` consecutive iterations (`c[i:i+W]`).
   - **reduction loop:** C/G may use `VREDSUM` and scalar `ADD acc`; V instead
     accumulates blocks with `VADD` and reduces lanes later. Select a final
     reduction route compatible with that partial-sum representation.
   - **map with a nested reduction:** a block is `W` consecutive output
     elements of the **same row**, accumulated as a vector. In matmul,
     `result[i][j:j+W] += a[i][k] * b[k][j:j+W]`; in convolucion,
     `result[i][j:j+W] += kernel[ki][kj] * imagen[i+ki][j+kj : j+kj+W]`. The
     scalar factor is copied to every lane with `VBCAST`; the vector factor is
     a piece of a row, so it is contiguous. Do not vectorize along a column
     (`b[k:k+W][j]` is not contiguous).

## Outputs

Check scalar tails explicitly. If no E is eligible, schedule them on C/G in
separate phases or reserve C/G for scalar work. If no scalar execution path
exists, reject unsupported partial vectors; do not pad or cross a row boundary.
Use scalar execution for the whole operation when it is the valid fallback.

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
    compute PE: send c back                         (VSEND)
    C/G: VREDSUM + ADD into acc; V: VADD into v4, VREDSUM later
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
