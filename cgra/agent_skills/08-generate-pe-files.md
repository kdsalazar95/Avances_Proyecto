# Skill: Generate PE Files

## Purpose

Write one text file per PE containing its instruction stream.

## Output Directory

Inside the folder of the program:

```text
programas/<operacion>/pe_instructions/
```

Create the directory if it does not exist.

## Output Files

One file per PE of the mesh, `PE{fila}{columna}.txt`, using the identifiers of
skill 06. For a `2x2` mesh:

```text
PE00.txt
PE01.txt
PE10.txt
PE11.txt
```

A `3x3` mesh needs `PE00.txt` through `PE22.txt`. A mesh with 11 or more rows or
columns (an index of 10 or more) pads each coordinate to the same width: in a `12x12` mesh, row 1 column
0 is `PE0100.txt` and row 10 column 0 is `PE1000.txt`.

## File Format

Each file must contain one instruction per execution cycle:

```text
// PE00 instruction list
// Mesh position: fila 0, columna 0
// One instruction per execution cycle.

00: MOV acc, 0.0
01: LD rA, a[0]
02: LD rB, b[0]
03: ADD rC, rA, rB
```

## Actions

1. Create `programas/<operacion>/pe_instructions/` if needed.
2. Write cycle-numbered instructions for each PE.
3. Keep cycle numbers aligned across all files.
4. Include `NOP` instructions for idle cycles so every file has the same cycle
   range.
5. Use in `LD`/`ST` only the bank names of the regions in `memoria.bin`
   (skill 02), with indices inside each region (`filas * columnas` words). The
   CGRA loads that memory as is and does not know the operation, so a bank that
   does not exist there fails at run time.

## Outputs

One text file per PE in the mesh (`filas * columnas` files in total).
