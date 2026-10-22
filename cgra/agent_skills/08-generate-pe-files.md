# Skill: Generate PE Files

## Purpose

Write one text file per PE containing its instruction stream.

Apply the [Mesh Compatibility Contract](mesh-compatibility.md). Small meshes
also get exactly filas * columnas files: 1x1 gets only PE00.txt, 1x2 gets
PE00.txt/PE01.txt, and 2x1 gets PE00.txt/PE10.txt. Do not invent extra border
PEs. Use the actual simulator map and permissions for every header.

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
// PE type: memoria (M) - NOP, MOV, LD, ST, SEND, RECV, VLD, VST, VSEND, VRECV
// One instruction per execution cycle.

00: LD rA, a[0]
01: SEND east, rA
02: LD rB, b[0]
03: SEND east, rB
```

The type line is a comment, taken from `src/cgra.json`: it documents which
instructions that file may contain.

## Actions

1. Create `programas/<operacion>/pe_instructions/` if needed.
2. Generate the complete instruction streams from the current manifest, then
   write cycle-numbered instructions for every PE, overwriting existing files
   in full (truncate; never append). Their existence or an old header is not a
   reason to skip generation. Remove only obsolete generated `PE*.txt` files
   from this operation's `pe_instructions/` whose names are outside the current
   mesh's expected set. This also covers a smaller mesh or a change in the
   coordinate padding width. Leave other files and operations alone.
3. Keep cycle numbers aligned across all files.
4. Include `NOP` instructions for idle cycles so every file has the same cycle
   range.
5. Write in each file only instructions of the PE's type (`src/cgra.json`):
   LD/ST and VLD/VST need memory capability (`M/G`); scalar arithmetic needs
   eligible `E/C/G`, and vector arithmetic needs eligible `V/C/G` plus vector
   hardware and vector mode. G can mix memory and compute instructions, one
   per cycle. Never emit SEND/RECV to itself for local execution.
6. Use in `LD`/`ST` (and `VLD`/`VST`) only the bank names of the regions in `memoria.bin`
   (skill 02), with indices inside each region (`filas * columnas` words); a
   `VLD`/`VST` at index `i` uses `i .. i + W - 1`, so all of them must be
   inside. The
   CGRA loads that memory as is and does not know the operation, so a bank that
   does not exist there fails at run time.
7. Keep explicit tile/phase and route records with the operation's generated
   manifest. Scalar and vector work sharing a C/G is ordered rather than
   duplicated in a cycle. Idle tile entries emit NOP unless the recorded plan
   gives that PE relay, shared-server or reduction work. An unavailable
   layout/route is an error, not a reason to publish an incomplete PE set.

## Outputs

One text file per PE in the mesh (`filas * columnas` files in total).
