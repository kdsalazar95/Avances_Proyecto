"""Agente RL del enlace epsilon-greedy: un LLM (GitHub Copilot CLI) elige el siguiente PE.

En cada paso del camino, con probabilidad ``epsilon``, el agente recibe el PE actual y sus
vecinos libres (con tipo, peso y posición) y responde con el identificador de uno de ellos.
El comando se ejecuta sin shell, con el prompt como argumento (``{prompt}`` en agente.json).
"""

import re
import subprocess
import tempfile

from malla import Malla

PATRON_PE = re.compile(r"PE\d+")


def construir_prompt(actual, libres, pesos, camino):
    """El prompt del paso: el PE actual, los vecinos libres y el camino recorrido hasta ahora."""
    filas = "\n".join(
        f"- {pe}: tipo {pesos[pe].tipo}, peso {pesos[pe].peso:.4g}, fila {Malla.posicion(pe)[0]}, "
        f"columna {Malla.posicion(pe)[1]}"
        for pe in libres)
    return (
        "Eres el agente de RL que elige el siguiente PE de un camino por una malla de una CGRA.\n"
        f"El camino va en {actual} y lleva {len(camino)} PEs. Elige uno de los vecinos libres (no visitados). "
        "Un peso menor es mejor (menos NOPs y tipo de PE más barato).\n"
        f"Vecinos libres:\n{filas}\n\n"
        f"Responde solo con el identificador de uno de ellos, por ejemplo {libres[0]}."
    )


def interpretar(salida, libres):
    """El primer identificador de PE de la respuesta que sea un vecino libre; ``None`` si no hay."""
    for pe in PATRON_PE.findall(salida):
        if pe in libres:
            return pe
    return None


class EleccionCopilot:
    """``eleccion(actual, libres, pesos, camino) -> PE`` (o ``None`` si la respuesta no sirve)."""

    def __init__(self, comando, timeout_s=300):
        if not comando or not any("{prompt}" in parte for parte in comando):
            raise ValueError("rl.comando debe incluir {prompt} (el prompt del paso)")
        self.comando = list(comando)
        self.timeout_s = timeout_s

    def __call__(self, actual, libres, pesos, camino):
        prompt = construir_prompt(actual, libres, pesos, camino)
        argumentos = [parte.replace("{prompt}", prompt) for parte in self.comando]
        try:
            proceso = subprocess.run(argumentos, capture_output=True, text=True, timeout=self.timeout_s,
                                     cwd=tempfile.gettempdir())
        except FileNotFoundError:
            raise RuntimeError(f"no se encontró el comando del agente RL: {self.comando[0]!r}")
        except subprocess.TimeoutExpired:
            return None
        if proceso.returncode != 0:
            raise RuntimeError(f"el agente RL terminó con código {proceso.returncode}: "
                               f"{(proceso.stderr or proceso.stdout).strip()[:500]}")
        return interpretar(proceso.stdout, libres)
