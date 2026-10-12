# Skill: Parse Program

## Purpose

Extract operations, memory accesses, loops, and dependencies from the LLVM IR
and the generated graphs.

## Inputs

- `programas/<operacion>/operaciones.ll`
- CFG dot graph
- DDG dot graph
- the size constants and the `memoria.bin` regions recorded in the manifest

## Actions

1. Read the LLVM IR and identify functions, basic blocks, branches, and loops.
   The work to schedule is in the function named after the operation
   (`@matmul`); `inicializar_<operacion>`, `guardar_memoria` and `escribir`
   only prepare and save the initial memory (skill 02), and `main` only picks
   the operation. Ignore the functions of the other operations.
2. Identify loop bounds from compare instructions such as `icmp slt`; they
   compare against the size constants of the source.
3. Identify induction variables and increments.
4. Identify memory operations:
   - `load`
   - `store`
   - `getelementptr`

   A 2D array appears as `getelementptr inbounds [N x [N x float]]` with two
   indices. In the CGRA every bank is flat, so translate it to a single index,
   `fila * ancho + columna`.

   Name each access after its region in `memoria.bin`, not after the C
   variable: `c` of `matmul` and `resultado` of `convolucion` are the bank
   `result` (see the table in skill 02).
5. Identify compute operations:
   - `fadd`, `add`
   - `fmul`, `mul`
   - `call float @llvm.fmuladd.f32(x, y, acc)` — clang fuses `acc += x * y` into
     this call, so **matmul and convolucion contain no `fmul` or `fadd` at
     all**. Read one `fmuladd` as a `MUL` followed by an `ADD`.
6. Ignore the initialization loops. They live in `inicializar_<operacion>()`,
   not in the operation's function: they fill the input arrays (with `srem` and `sitofp` for
   expressions such as `(float)((i + j) % 5 + 1)`) and their values already are
   in `memoria.bin`, so they are **not** work to schedule. The operation's
   function only calls `inicializar_<operacion>` and then runs the computation
   loops.
7. Read the DDG to identify true data dependencies, and distinguish scalar
   loop-control dependencies from computation dependencies.

## Loop Count per Operation

Use this as a sanity check that nothing was missed:

| Operation | Loops in `<operacion>()` (all computation) | Loops in `inicializar_<operacion>()` |
|---|---|---|
| `reduccion` | 2 | 1 |
| `matmul` | 3 | 2 |
| `convolucion` | 4 | 6 |

## Outputs

Produce an intermediate representation of the computation loops. For
`reduccion` with `TAMANO_VECTOR = 12`:

```text
loops:
  loop_map:
    lower_bound=0
    upper_bound=12
    body:
      load a[i]
      load b[i]
      add
      store c[i]

  loop_reduction:
    lower_bound=0
    upper_bound=12
    body:
      load c[i]
      add into result
      store result

dependencies:
  c[i] depends on a[i], b[i]
  result iteration i depends on result iteration i-1
```

For `matmul` with `TAMANO_MATRIZ = N`:

```text
loops:
  loop_i, loop_j:         independent output elements C[i][j]
  loop_k:
    lower_bound=0
    upper_bound=N
    body:
      load a[i*N + k]
      load b[k*N + j]
      fmuladd -> mul + add into acc
    store result[i*N + j]

dependencies:
  acc of (i, j) is carried across k only
```

For `convolucion` with image `I`, kernel `K` and output `S = I - K + 1`:

```text
loops:
  loop_i, loop_j:         independent output elements result[i][j]
  loop_ki, loop_kj:
    body:
      load imagen[(i+ki)*I + (j+kj)]
      load kernel[ki*K + kj]
      fmuladd -> mul + add into acc
    store result[i*S + j]

dependencies:
  acc of (i, j) is carried across (ki, kj) only
```

## Notes

If a reduction is present, capture it explicitly. A scalar value loaded,
updated, and stored on each iteration is a loop-carried dependency: in
`reduccion` it spans the whole vector, while in `matmul` and `convolucion` it is
local to one output element.
