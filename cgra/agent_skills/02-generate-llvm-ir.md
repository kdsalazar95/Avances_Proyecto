# Skill: Generate LLVM IR and Initial Memory

## Purpose

Generate LLVM IR from `operaciones/operaciones.c`, and run it once for the
requested operation to produce the initial memory of the CGRA, `memoria.bin`.

## Required Tool

`clang-18`

## Commands

Run all of them from the operation's folder, `programas/<operacion>/` (create it
if needed):

```sh
clang-18 -S -emit-llvm ../../operaciones/operaciones.c -o operaciones.ll
clang-18 ../../operaciones/operaciones.c -o operaciones.out && ./operaciones.out <operacion>
```

The first command produces the IR. It holds every operation, each in the
function named after it (`@reduccion`, `@matmul`, `@convolucion`). The second
compiles the source to a native executable and runs it with the operation's
name: its `inicializar_<operacion>()` creates and initializes the arrays and
saves them to `memoria.bin` in the current directory, with `guardar_memoria()`
from `operaciones/memoria_cgra.h`.

## The Initial Memory

Like the memory of a processor, `memoria.bin` is binary: a flat array of
`float32` words preceded by a symbol table. Each symbol is one region (a bank):
its name, its first word, and its shape in `filas x columnas`. Matrices are
stored row by row. The exact layout is documented in
`operaciones/memoria_cgra.h`.

The region names are the bank names that the PE programs use in `LD`/`ST`
(skill 08). They come from the `RegionMemoria` table at the end of
`inicializar_<operacion>()`, not from the C variable names: in `matmul` the
array `c` is saved as `result`, and in `convolucion` `resultado` is saved as
`result`.

| Operation | Regions in `memoria.bin` |
|---|---|
| `reduccion` | `a`, `b`, `c` (1 x `TAMANO_VECTOR`), `result` (1 x 1) |
| `matmul` | `a`, `b`, `result` (`TAMANO_MATRIZ` x `TAMANO_MATRIZ`) |
| `convolucion` | `imagen`, `kernel`, `result` (`TAMANO_SALIDA` x `TAMANO_SALIDA`) |

The CGRA never generates or rewrites this memory by itself; `run_cgra.py` only
loads it (and saves the results back with `--write-back`).

## Actions

1. Run the commands from `programas/<operacion>/`, unless the user specified
   another output directory.
2. Stop and report the compiler error if either compilation fails, or the
   program's error if it cannot write `memoria.bin`. An unknown operation name
   makes the executable print the list of operations and exit with an error.
3. Check that the program printed `Memoria inicial guardada en memoria.bin` and
   record the regions it saved.
4. Delete `operaciones.out`; only `memoria.bin` is needed afterwards.
5. Use the generated `.ll` for all later stages.

## Outputs

```text
programas/<operacion>/operaciones.ll
programas/<operacion>/memoria.bin
```

## Notes

The generated IR is the canonical representation for graph generation and
scheduling. Do not hand-edit it before passing it to `opt-18`.

The IR also contains `main`, the other operations, every
`inicializar_<operacion>()` and the helpers from `memoria_cgra.h`
(`guardar_memoria`, `escribir`). None of them is work for the CGRA: only the
loops of the requested operation's function are scheduled.
