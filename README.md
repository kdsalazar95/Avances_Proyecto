# Proyecto

Dos sistemas para optimizar la configuración de una CGRA heterogénea, que se contrastan entre sí. Usan las
mismas entradas, la misma referencia 32x32, el mismo simulador y la misma función de costo (con los mismos
normalizadores); cambian el agente y el bloque que genera los programas de cada PE. Cada sistema vive en su rama.

| Rama | Carpeta | Agente | Evaluador (genera los PE*.txt) |
|---|---|---|---|
| `feature/sistema1-llm` (esta) | `rl/` | LLM, sin Greedy | Skills 01–09 (LLM) |
| `feature/sistema2-greedy` | `sistema2_greedy/` | ε-greedy, sin LLM | Dijkstra |

`cgra/` (simulador, `operaciones.c` y skills) es común a los dos.

Sistema 1: consulte [su README](rl/README.md). Barrido de epsilon con los datos para la comparación:

```bash
uv run python -m rl barrido --config rl/config/comparacion.json --epsilons 0 10 20 80 90 100
```

Pruebas:

```bash
python3 cgra/test/run_tests.py                         # simulador
uv run python -m unittest discover -s rl/tests -t .    # Sistema 1
```
