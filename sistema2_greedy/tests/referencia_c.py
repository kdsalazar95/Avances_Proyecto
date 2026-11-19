"""Referencia de las pruebas: lo que calcula el propio C, sin escribir cada
operación otra vez a mano.

``Operaciones(fuente)`` copia un ``operaciones.c`` (y los ``.h`` de su
carpeta) a una carpeta temporal, corre el skill 02 para cada operación de su
tabla OPERACIONES (``operaciones.ll`` y ``memoria.bin``) y, con el arnés
``memoria_final.c``, guarda la memoria que deja el C al terminar cada
operación, en float32. Sirve para cualquier operación: las pruebas no saben
cuáles son.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from dijkstra.compilar import CLANG, preparar
from memoria import ImagenMemoria
from run_cgra import SimuladorCGRA

ARNES = Path(__file__).with_name("memoria_final.c")
CARPETA_DATOS = Path(__file__).with_name("datos")
OTRAS_OPERACIONES = CARPETA_DATOS / "otras_operaciones.c"
TOLERANCIA = 1e-5  # el grafo y el simulador calculan en float de Python; el C, en float32


class Operaciones:
    """Las operaciones de un ``operaciones.c`` compiladas en una carpeta temporal."""

    def __init__(self, fuente, cabeceras=None):
        fuente = Path(fuente)
        self._temporal = tempfile.TemporaryDirectory()
        raiz = Path(self._temporal.name)
        codigo = raiz / "operaciones"
        codigo.mkdir()
        for cabecera in sorted(Path(cabeceras or fuente.parent).glob("*.h")):
            shutil.copy(cabecera, codigo)
        self.fuente = codigo / "operaciones.c"
        shutil.copy(fuente, self.fuente)
        self.programas = raiz / "programas"
        self.nombres = SimuladorCGRA.operaciones(self.fuente)
        preparar(self.nombres, self.programas, self.fuente)

        arnes = raiz / "memoria_final.out"
        subprocess.run([CLANG, "-finstrument-functions", "-I", str(codigo), '-DFUENTE="operaciones.c"',
                        str(ARNES), "-o", str(arnes)], check=True, capture_output=True)
        self.finales = {}
        with tempfile.TemporaryDirectory(dir=raiz) as trabajo:
            for operacion in self.nombres:
                subprocess.run([str(arnes), operacion], cwd=trabajo, check=True, capture_output=True)
                self.finales[operacion], _ = ImagenMemoria.leer(Path(trabajo) / "memoria_final.bin").regiones()

    def carpeta(self, operacion):
        return self.programas / operacion

    def memoria(self, operacion):
        """``memoria.bin`` de la operación."""
        return self.carpeta(operacion) / "memoria.bin"

    def inicial(self, operacion):
        """``({región: valores}, {región: (filas, columnas)})`` de la memoria inicial."""
        return ImagenMemoria.leer(self.memoria(operacion)).regiones()

    def final(self, operacion):
        """``{región: valores}`` que deja el C al terminar la operación."""
        return self.finales[operacion]

    def cerrar(self):
        self._temporal.cleanup()


def diferencias(esperado, obtenido, tolerancia=TOLERANCIA):
    """Posiciones en que ``obtenido`` no coincide con ``esperado`` (las regiones de ``esperado``)."""
    distintas = []
    for region, valores in esperado.items():
        otros = obtenido.get(region)
        if otros is None or len(otros) != len(valores):
            distintas.append(f"{region}: forma distinta")
            continue
        for posicion, (e, o) in enumerate(zip(valores, otros)):
            if not math.isclose(e, o, rel_tol=tolerancia, abs_tol=tolerancia):
                distintas.append(f"{region}[{posicion}]: C = {e}, obtenido = {o}")
    return distintas
