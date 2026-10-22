# Skill: Validate Schedule

## Purpose

Check that the generated per-PE instruction files are internally consistent and
respect mesh communication.

## Checks

Apply the [Mesh Compatibility Contract](mesh-compatibility.md), including its
acceptance cases. The small-mesh layouts are resolved by `FormatoCGRA`; each
generated program is still validated against the shared loader below.

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
10. The final reduction result is stored by `output_pe` from the manifest.
    Prefer PE00 when reachable and memory-capable; local G stores directly.
    A different endpoint is valid when the capability/routing plan requires it
    or the user requested it. Check that all contributing partials reach it.
11. Every `LD`/`ST` names a region of `memoria.bin` and stays inside its size;
    the `W` words of a `VLD`/`VST` too.
    Read the regions from its symbol table (layout in `operaciones/memoria_cgra.h`), or
    load it with `ImagenMemoria.leer()` of `src/memoria.py`.
12. Validate actual instruction permissions: memory accesses on M/G, scalar
    arithmetic on E/C/G, vector arithmetic on V/C/G when configured. G can
    combine both roles; C does not implicitly gain memory access. No scalar
    arithmetic on V or vector arithmetic/relays on E. In scalar mode no vector
    instruction is emitted.
13. `memoria.bin` fits in `palabras_memoria` of `src/cgra.json`.
14. Every assigned compute PE has compatible local accesses or outbound/return
    routes to its memory server. Check vertical and turning routes too; a
    geometric path alone is insufficient for vector transfers.
15. No work-list, tile-size or round formula divides/modulos by an empty PE
    set. Full vectors stay in a bank and an output row. Tails go to scalar
    capability; no implicit padding or unimplemented lane packing.
16. Each output accumulator resets before reuse in a new tile, while input
    blocks of a reduction preserve their local sum. Shared C/G scalar/vector
    phases do not overlap. Validate partial final rounds and NOPs except for
    required relays, shared-server or reduction work.
17. The manifest, per-PE headers and actual `FormatoCGRA` map agree. Report
    unsupported implementation paths and impossible hardware mappings
    accurately; never count a documentation-only policy as implemented support.

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

- **reduccion:** each selected compute PE accumulates its assigned values
  before combining partials using a compatible scalar/vector plan. A single
  contributor skips inter-compute reduction. The output memory-capable PE
  stores the final sum; on a local G no network transfer is required.
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
