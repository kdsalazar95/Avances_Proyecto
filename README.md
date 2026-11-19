# Proyecto

Dos sistemas para optimizar la configuración de una CGRA heterogénea, que se contrastan entre sí. Usan las
mismas entradas, la misma referencia 32x32, el mismo simulador y la misma función de costo (con los mismos
normalizadores); cambian el agente y el bloque que genera los programas de cada PE. Cada sistema vive en su rama.

| Rama | Carpeta | Agente | Evaluador (genera los PE*.txt) |
|---|---|---|---|
| `feature/sistema1-llm` | `rl/` | LLM, sin Greedy | Skills 01–09 (LLM) |
| `feature/sistema2-greedy` (esta) | `sistema2_greedy/` | ε-greedy, sin LLM | Dijkstra |

`cgra/` (simulador, `operaciones.c` y skills) es común a los dos.

Sistema 2: consulte [su README](sistema2_greedy/README.md). Barrido de epsilon con los datos para la comparación
(desde `sistema2_greedy/`):

```bash
python3 herramientas/barrido.py --epsilons 0 10 20 80 90 100
```

Pruebas:

```bash
python3 cgra/test/run_tests.py              # simulador
python3 sistema2_greedy/tests/run_tests.py  # Sistema 2
```
