# Proyecto

Dos sistemas para optimizar la configuración de una CGRA heterogénea, que se contrastan entre sí.
Usan las mismas entradas, el mismo simulador y la misma función de costo; cambian el agente y el
bloque que genera los programas de cada PE.

| Carpeta | Agente | Evaluador (genera los PE*.txt) |
|---|---|---|
| `rl/` (Sistema 1) | LLM, sin Greedy | Skills 01-09 (LLM) |
| `sistema2_greedy/` (Sistema 2) | ε-greedy, sin LLM | Dijkstra (sustituye los skills 04-07) |
| `cgra/` (común) | simulador, `operaciones.c` y skills, compartidos por los dos | |

- Sistema 1: consulte [su README](rl/README.md).
- Sistema 2: consulte [su README](sistema2_greedy/README.md).

Correr el Sistema 2 (desde `sistema2_greedy/`):

```bash
python3 agente_greedy.py      # resultados en sistema2_greedy/resultados/greedy/
```

Pruebas:

```bash
python3 cgra/test/run_tests.py              # simulador
python3 sistema2_greedy/tests/run_tests.py  # Sistema 2
uv run python -m unittest discover -s rl/tests -t .   # Sistema 1
```
