# Proyecto

- `cgra/`: simulador, operaciones y skills, compartidos con la solución RL.
- `dijkstra_greedy/`: la solución Dijkstra + Greedy, el ciclo agente Greedy ↔ evaluador. Dijkstra
  sustituye los skills 04-07. Consulte [su README](dijkstra_greedy/README.md).

Correr la búsqueda (desde `dijkstra_greedy/`):

```bash
python3 agente_greedy.py      # resultados en dijkstra_greedy/resultados/greedy/
```

Pruebas:

```bash
python3 cgra/test/run_tests.py             # simulador
python3 dijkstra_greedy/tests/run_tests.py  # Dijkstra + Greedy
```
