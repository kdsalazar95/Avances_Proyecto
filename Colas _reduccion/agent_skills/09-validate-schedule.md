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
10. The final reduction result is stored by the memory PE at the end of the
    route of skill 06 (`PE00` with the default file), unless the user
    requested a different output PE.
11. Every `LD`/`ST` names a region of `memoria.bin` and stays inside its size.
    Read the regions from its symbol table (layout in `src/memoria_cgra.h`), or
    load it with `cargar_memoria()` of `src/memoria_binaria.py`.
12. Every PE only uses instructions of its type in `src/cgra.json`: no `LD`/`ST`
    in a compute PE, no `ADD`/`SUB`/`MUL`/`DIV` in a memory PE.
13. `memoria.bin` fits in `palabras_memoria` of `src/cgra.json`.

Checks 1-6 and 11-13 are what `run_cgra.py` enforces when it loads the program;
run them with the same code instead of re-implementing them:

```sh
cd src && python3 -c "
from cgra import CGRA
from cargador import cargar_programa
from formato_cgra import cargar_formato
cargar_programa(CGRA(cargar_formato()), '<programa>/memoria.bin', '<programa>/pe_instructions')
print('Carga valida')"
```

Then run `uv run src/run_cgra.py src/<programa>` and compare the result with
what the `.c` computes (checks 7-10).

## Per Operation

- **reduccion:** each compute PE accumulates its own values before the
  reduction route, and the final value is stored by a memory PE.
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
