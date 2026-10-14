# Skill: Schedule Instructions

## Purpose

Convert mapped PE work into cycle-by-cycle instructions.

## Instruction Set

```text
MOV  reg, imm
LD   reg, mem[index]
ST   mem[index], reg
ADD  dst, src1, src2
SUB  dst, src1, src2
MUL  dst, src1, src2
DIV  dst, src1, src2
SEND dir, reg
RECV dir, reg
NOP
```

## Instructions per PE Type

The instructions a PE may emit depend on its type (`src/cgra.json`, skill 01):

```text
memoria (M): NOP, MOV, LD, ST, SEND, RECV
computo (C): NOP, MOV, ADD, SUB, MUL, DIV, SEND, RECV
```

Never emit an instruction outside the type of the PE: `run_cgra.py` refuses to
load the program and reports the PE and cycle.

## Scheduling Rules

1. Emit exactly one instruction per PE per cycle.
2. Insert `NOP` when a PE has no work during a cycle.
3. Respect the PE types (above): `LD`/`ST` only on memory PEs, arithmetic only
   on compute PEs.
4. Do not use a register before the instruction that defines it, and do not
   overwrite it before its last use.
5. Align neighbor communication: a `SEND <dir>` and the neighbor's
   `RECV <opposite>` go in the **same cycle**. The data is available to the
   receiver in that cycle.
6. Preserve operation dependencies from the DDG.
7. For local accumulation, accumulate in the compute PE before starting the
   reduction across the mesh.

The tables below are for a compute PE **next to its server** that is the only
compute PE of that server, as with two compute columns (skill 06). `<to C>` is
the direction from the server to its compute PE (`east` for a west server,
`west` for an east server) and `<to M>` is the opposite one. All server/compute
pairs run the same table at the same time. With more compute columns, see
[Shared or Distant Servers](#shared-or-distant-servers).

## Reduccion Schedule

1. Map and local accumulation, one **6-cycle step** per iteration `i` of the
   compute PE (skill 06). The step is pipelined: the compute PE finishes the
   previous iteration while its server already loads the next one.

   ```text
   cycle | server (M)                    | compute (C)
   ------+-------------------------------+------------------------------------
   t+0   | LD   rA, a[i]                 | ADD  acc, acc, rC   (previous i)
   t+1   | SEND <to C>, rA               | RECV <to M>, rA
   t+2   | LD   rB, b[i]                 | NOP
   t+3   | SEND <to C>, rB               | RECV <to M>, rB
   t+4   | ST   c[previous i], rC        | ADD  rC, rA, rB
   t+5   | RECV <to C>, rC               | SEND <to M>, rC
   ```

   - In the first step, the compute PE emits `MOV acc, 0.0` at `t+0` and the
     server emits `NOP` at `t+4` (there is no previous iteration).
   - After the last step, one closing cycle: the server emits
     `ST c[last i], rC` and the compute PE emits `ADD acc, acc, rC`.
   - Pairs with fewer iterations than the busiest one pad with `NOP` after their
     closing cycle, so all pairs reach the reduction route together.

2. Reduction route (skill 06), two cycles per hop between compute PEs:

   ```text
   cycle t:     sender   SEND <dir>, acc     receiver RECV <opposite>, rT
   cycle t+1:   sender   NOP                 receiver ADD acc, acc, rT
   ```

   All rows perform their hop at the same time; the column hops come after, and
   use `SEND north` / `RECV south`.

3. Store, two cycles: the compute PE at the end of the route sends the total to
   its server (`SEND <to M>, acc` / `RECV <to C>, rC`), and the server emits
   `ST result[0], rC` in the next cycle.

## Matmul and Convolucion Tile Schedule

Schedule one output tile at a time, with the mapping of skill 06. Every step of
the accumulation (`k` in matmul, `(ki, kj)` in convolucion) is a **4-cycle
step**, pipelined so that the server loads and sends every cycle while the
compute PE multiplies and accumulates the previous step:

```text
cycle | server (M)                  | compute (C)
------+-----------------------------+------------------------------------
t+0   | LD   rA, <operand A>        | MUL  rC, rA, rB      (previous step)
t+1   | SEND <to C>, rA             | RECV <to M>, rA
t+2   | LD   rB, <operand B>        | ADD  acc, acc, rC    (previous step)
t+3   | SEND <to C>, rB             | RECV <to M>, rB
```

- In the first step of a tile, the compute PE emits `NOP` at `t+0` and
  `MOV acc, 0.0` at `t+2`.
- The operands are the ones of skill 06 (`a`/`b` in matmul, `imagen`/`kernel`
  in convolucion).

After the last step, close the tile in 4 cycles:

```text
cycle | server (M)                  | compute (C)
------+-----------------------------+------------------------------------
e+0   | NOP                         | MUL  rC, rA, rB
e+1   | NOP                         | ADD  acc, acc, rC
e+2   | RECV <to C>, rC             | SEND <to M>, acc
e+3   | ST   result[...], rC        | NOP
```

A pair whose compute PE falls outside the output in an edge tile emits `NOP` on
both PEs for the whole tile (`4 * steps + 4` cycles). Then move to the next tile
and start again.

A tile therefore lasts `4 * steps + 4` cycles: `4 * 9 + 4 = 40` in a `9x9`
matmul and `4 * 16 + 4 = 68` in a convolution with a `4x4` kernel.

## Shared or Distant Servers

With more than two compute columns, a memory PE serves several compute PEs and
some of them are at distance `d > 1` (skill 06). There is no fixed table for
them; build the cycles from the rules above:

1. A delivery to a compute PE at distance `d` takes `d` cycles, one hop per
   cycle along the row. Each compute PE in between receives the value into `rT`
   in one cycle (`RECV <to M>, rT`) and sends it on in the next
   (`SEND <to C>, rT`). The last hop receives into the real register (`rA`,
   `rB`), and results travel back to the server the same way.
2. A relaying compute PE can only relay in a cycle where it is not using that
   link, and `rT` must be free: it holds one value in transit at a time.
3. A server with several compute PEs serves them one after another: it loads
   and sends the operands of one, then of the next. The nearest one can keep
   using the table above; the farther ones take longer steps because of the
   extra hops.
4. All pairs must reach the reduction route (or the end of the tile) together:
   the shorter ones pad with `NOP`.

Write the resulting step once as a table like the ones above, check it with
skill 09, and only then repeat it for every iteration or tile.

## Default Registers

```text
rA   first loaded operand
rB   second loaded operand
rC   computed element result
rT   received temporary value (reduction route, relays)
acc  local accumulation
```

## Output Format

Produce a global schedule table first when useful, with one column per PE of the
configured mesh and its type, in the order of the map printed in skill 01:

```text
Cycle | PE00 (M)          | PE01 (C)          | ... | PE0<last> (M)
------+-------------------+-------------------+-----+-------------------
00    | LD rA, a[0]       | MOV acc, 0.0      | ... | LD rA, a[...]
01    | SEND east, rA     | RECV west, rA     | ... | SEND west, rA
```

For a large mesh the table is impractical: go straight to the per-PE instruction
streams and pass them to skill 08.
