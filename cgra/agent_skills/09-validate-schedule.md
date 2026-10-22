# Skill: Validate Schedule

## Purpose

Check that the generated per-PE instruction files are internally consistent and
respect mesh communication.

## Checks

1. All `filas * columnas` PE files exist, one per PE of the mesh.
   Compare the complete set of generated `PE*.txt` filenames with the expected
   IDs: no obsolete PE files may remain. `Programa.leer()` reads only expected
   IDs, so successful loading alone does not detect obsolete extra files.
2. All files cover the same cycle range.
3. Each cycle has exactly one instruction per PE.
4. Every `SEND` has a matching neighbor `RECV`, and every `VSEND` a matching
   `VRECV` (a `VSEND` never pairs with a `RECV`).
5. Direction pairs are valid:
   - `SEND west` matches neighbor `RECV east`
   - `SEND east` matches neighbor `RECV west`
   - `SEND north` matches neighbor `RECV south`
   - `SEND south` matches neighbor `RECV north`
6. No diagonal communication is used, and no transfer leaves the mesh.
7. Registers, scalar and vector, are defined before use.
8. Stores occur only after their source values are computed.
9. All loop iterations are scheduled exactly once. In `vectorial` mode, every
   element is in exactly one block (or segment) or in the tail, never in both.
10. The final reduction result is stored by the memory PE at the end of the
    route of skill 06 (`PE00`), unless the user
    requested a different output PE.
11. Every `LD`/`ST` names a region of `memoria.bin` and stays inside its size;
    the `W` words of a `VLD`/`VST` too.
    Read the regions from its symbol table (layout in `operaciones/memoria_cgra.h`), or
    load it with `ImagenMemoria.leer()` of `src/memoria.py`.
12. Every PE only uses instructions of its type in `src/cgra.json`: no `LD`/`ST`
    in a compute PE, no `ADD`/`SUB`/`MUL`/`DIV` in a memory PE, and the same for
    the vector forms: `VLD`/`VST` only in memory PEs (`M`), vector arithmetic only
    in `V` and `C`, no scalar arithmetic in `V` and no vector one in `E`. In
    `escalar` mode there is no vector instruction at all.
13. `memoria.bin` fits in `palabras_memoria` of `src/cgra.json`.

Checks 1-6 and 11-13 are what `run_cgra.py` enforces when it loads the program;
run them with the same code instead of re-implementing them:

```sh
cd src && python3 -c "
from cargador import Cargador
from cgra import CGRA
from formato_cgra import FormatoCGRA
Cargador(CGRA(FormatoCGRA.cargar())).cargar('../programas/<operacion>/memoria.bin', '../programas/<operacion>/pe_instructions')
print('Carga valida')"
```

Then run `uv run src/run_cgra.py <operacion>` and compare the result with what
`<operacion>()` of `operaciones/operaciones.c` computes (checks 7-10).

## Per Operation

- **reduccion:** each compute PE accumulates its own values before the
  reduction route, and the final value is stored by a memory PE.
- **matmul and convolucion:** every output element is stored exactly once, all
  tiles are processed, and the PEs outside the output in an edge tile only emit
  `NOP`. Each operand reaches its compute PE, from that PE's server, before the
  `MUL` that uses it.

## Output

Report either:

```text
Schedule validation passed.
```

or a concise list of validation errors with file names and cycle numbers.

## Notes

If floating-point tree reduction is used, report that it may differ from the
strict scalar loop order due to rounding.
