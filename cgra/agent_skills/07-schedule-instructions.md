# Skill: Schedule Instructions

## Purpose

Convert mapped PE work into cycle-by-cycle instructions.

## Instruction Set

```text
MOV  reg, imm
LD   reg, mem[index]
ST   mem[index], reg
ADD  dst, src1, src2
MUL  dst, src1, src2
SEND dir, reg
RECV dir, reg
NOP
```

## Scheduling Rules

1. Emit exactly one instruction per PE per cycle.
2. Insert `NOP` when a PE has no work during a cycle.
3. Do not use a register before the instruction that defines it.
4. Align neighbor communication:
   - if one PE emits `SEND west, acc`, the west neighbor should emit
     `RECV east, rT` in the same cycle or in the architecture-defined receive
     cycle.
5. Preserve operation dependencies from the DDG.
6. For local accumulation, accumulate values produced by the same PE before
   starting mesh reduction.

## Reduccion Schedule

1. `MOV acc, 0.0` on every PE.
2. Map phase, for each index `i` owned by the PE:

   ```text
   LD  rA, a[i]
   LD  rB, b[i]
   ADD rC, rA, rB
   ST  c[i], rC
   ```

3. Local accumulation, for each index `i` owned by the PE:

   ```text
   LD  rC, c[i]
   ADD acc, acc, rC
   ```

4. Reduction route (skill 06), two cycles per hop:

   ```text
   cycle t:     sender   SEND west, acc      receiver RECV east, rT
   cycle t+1:   sender   NOP                 receiver ADD acc, acc, rT
   ```

   All rows perform their hop at the same time; the column hops come after, and
   use `SEND north` / `RECV south`.
5. `ST result[0], acc` on `PE00`.

PEs that own fewer indices than the busiest one pad their map and accumulation
phases with `NOP`, so all files stay aligned.

## Matmul and Convolucion Tile Schedule

Schedule one output tile at a time, with the mapping of skill 06:

1. `MOV acc, 0.0` on every PE that owns an in-range element of this tile.
2. For every step of the accumulation (`k` in matmul, every `(ki, kj)` of the
   window in convolucion):
   - the PE that brings a value into the mesh emits its `LD`, and the PEs that
     are not involved in that cycle emit `NOP`;
   - the value advances one hop per cycle: the sender emits `SEND <dir>, <reg>`
     and the receiver emits `RECV <opuesta>, <reg>` in the same cycle. Crossing
     a row takes `columnas - 1` hops and crossing a column takes `filas - 1`;
   - in convolucion, each PE also loads its own pixel with `LD rA`;
   - with both operands in place, every PE emits `MUL rC, rA, rB` and
     `ADD acc, acc, rC`.
3. `ST result[...], acc` once the tile's accumulation is complete.
4. `NOP` for the PEs that fall outside the output in an edge tile, for as many
   cycles as the tile lasts.
5. Move to the next tile and start again from `MOV acc, 0.0`.

A value that travels costs one cycle per hop, so a step costs one `LD` plus the
hops plus `MUL` and `ADD`. In matmul both operands travel; in convolucion only
the kernel coefficient does, because every PE needs a different pixel.

## Default Registers

```text
rA   first loaded operand
rB   second loaded operand
rC   computed element result
rT   received temporary value
acc  local accumulation
```

## Output Format

Produce a global schedule table first when useful, with one column per PE of the
configured mesh (shown here for a `2x2` mesh):

```text
Cycle | PE00              | PE01              | PE10              | PE11
------+-------------------+-------------------+-------------------+-------------------
00    | MOV acc, 0.0      | MOV acc, 0.0      | MOV acc, 0.0      | MOV acc, 0.0
```

For a large mesh the table is impractical: go straight to the per-PE instruction
streams and pass them to skill 08.
