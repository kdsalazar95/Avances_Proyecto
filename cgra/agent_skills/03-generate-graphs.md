# Skill: Generate Graphs

## Purpose

Generate control-flow and data-dependency graphs from LLVM IR.

## Required Tool

`opt-18`

## Commands

```sh
opt-18 -passes=dot-cfg -disable-output operaciones.ll
opt-18 -passes=dot-ddg -disable-output operaciones.ll
```

## Actions

1. Run both commands from `programas/<operacion>/`, where skill 02 left
   `operaciones.ll`, so the graphs of each operation stay apart.
2. Locate the CFG dot file of the operation's function: `.<operacion>.dot`
   (for example `.matmul.dot`).
3. Locate the DDG dot file of the same function: `ddg.<operacion>..dot`
   (for example `ddg.matmul..dot`).
4. Preserve generated dot files; later skills should read them directly.

## Outputs

```text
programas/<operacion>/.<operacion>.dot
programas/<operacion>/ddg.<operacion>..dot
```

## Notes

The passes write one graph per function of `operaciones.c`: also `main`, the
other operations, every `inicializar_<operacion>`, `guardar_memoria` and
`escribir`. Later skills only use the graphs of the requested operation; the
others are other operations or describe how the initial memory is prepared,
not work for the CGRA.

LLVM graph pass filenames may vary with function names. Find the generated
`.dot` files of the operation's function rather than hard-coding the names.
