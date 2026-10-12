# Skill: Validate Schedule

## Purpose

Check that the generated per-PE instruction files are internally consistent and
respect mesh communication.

## Checks

1. All `filas * columnas` PE files exist, one per PE of the mesh.
2. All files cover the same cycle range.
3. Each cycle has exactly one instruction per PE.
4. Every `SEND` has a matching neighbor `RECV`.
5. Direction pairs are valid:
   - `SEND west` matches neighbor `RECV east`
   - `SEND east` matches neighbor `RECV west`
   - `SEND north` matches neighbor `RECV south`
   - `SEND south` matches neighbor `RECV north`
6. No diagonal communication is used, and no transfer leaves the mesh.
7. Registers are defined before use.
8. Stores occur only after their source values are computed.
9. All loop iterations are scheduled exactly once.
10. The final reduction result is stored by `PE00`, unless the user requested a
    different output PE.
11. Every `LD`/`ST` names a region of `memoria.bin` and stays inside its size.
    Read the regions from its symbol table (layout in `operaciones/memoria_cgra.h`), or
    load it with `ImagenMemoria.leer()` of `src/memoria.py`.

## Per Operation

- **reduccion:** each PE accumulates its own values before the reduction route.
- **matmul and convolucion:** every output element is stored exactly once, all
  tiles are processed, and the PEs outside the output in an edge tile only emit
  `NOP`. Each value that travels reaches every PE of its row, column or the
  whole mesh before the `MUL` that uses it.

## Output

Report either:

```text
Schedule validation passed.
```

or a concise list of validation errors with file names and cycle numbers.

## Notes

If floating-point tree reduction is used, report that it may differ from the
strict scalar loop order due to rounding.
