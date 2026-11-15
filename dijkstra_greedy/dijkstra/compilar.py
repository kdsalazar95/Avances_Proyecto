"""Skills 02 y 03 desde Python: IR, memoria inicial y grafos de cada operación.

Son los comandos de ``agent_skills/02-generate-llvm-ir.md`` y
``agent_skills/03-generate-graphs.md``, sin cambios (Dijkstra solo sustituye
los skills 04-07). El ciclo agente-evaluador (``agente_greedy.py``) los corre al
empezar:

    cd programas/<operacion>
    clang-18 -S -emit-llvm ../../operaciones/operaciones.c -o operaciones.ll      # skill 02
    clang-18 ../../operaciones/operaciones.c -o operaciones.out && ./operaciones.out <operacion>
    rm operaciones.out
    opt-18 -passes=dot-cfg -disable-output operaciones.ll                        # skill 03
    opt-18 -passes=dot-ddg -disable-output operaciones.ll

Todo depende solo de ``operaciones.c`` (y de ``memoria_cgra.h``), no de la
configuración de la CGRA: se genera una vez y sirve para todas las mallas. Solo
se vuelve a generar si falta algo o si el código fuente es más nuevo.
"""

import os
import subprocess
from pathlib import Path

from . import CARPETA_PROGRAMAS, FUENTE_OPERACIONES

CLANG = "clang-18"
OPT = "opt-18"
FUENTE = FUENTE_OPERACIONES
SALIDAS = ("operaciones.ll", "memoria.bin")  # skill 02


def grafos(operacion):
    """Los grafos del skill 03 de la función de la operación: CFG y DDG."""
    return (f".{operacion}.dot", f"ddg.{operacion}..dot")
MENSAJE_MEMORIA = "Memoria inicial guardada en memoria.bin"


class ErrorCompilacion(RuntimeError):
    """Falló el skill 02 (clang, el programa o el memoria.bin que debía escribir) o el 03 (opt)."""


def desactualizada(operacion, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE):
    """Verdadero si a la operación le falta un archivo de los skills 02-03 o alguno es
    más viejo que operaciones.c o que los .h de su carpeta."""
    fuente = Path(fuente)
    salidas = [Path(carpeta) / operacion / nombre for nombre in (*SALIDAS, *grafos(operacion))]
    if not all(s.exists() for s in salidas):
        return True
    cambio = max(f.stat().st_mtime for f in [fuente, *fuente.parent.glob("*.h")])
    return min(s.stat().st_mtime for s in salidas) < cambio


def compilar(operacion, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE, clang=CLANG, opt=OPT):
    """Corre los comandos de los skills 02 y 03 en ``carpeta/<operacion>/``."""
    destino = Path(carpeta) / operacion
    destino.mkdir(parents=True, exist_ok=True)
    relativa = os.path.relpath(fuente, destino)
    ejecutable = destino / "operaciones.out"
    try:
        _correr([clang, "-S", "-emit-llvm", relativa, "-o", "operaciones.ll"], destino)
        _correr([clang, relativa, "-o", ejecutable.name], destino)
        salida = _correr([f"./{ejecutable.name}", operacion], destino)
    finally:
        ejecutable.unlink(missing_ok=True)
    if MENSAJE_MEMORIA not in salida:
        raise ErrorCompilacion(f"{operacion}: el programa no escribió memoria.bin:\n{salida}")
    for pase in ("dot-cfg", "dot-ddg"):
        _correr([opt, f"-passes={pase}", "-disable-output", "operaciones.ll"], destino)
    faltan = [nombre for nombre in grafos(operacion) if not (destino / nombre).is_file()]
    if faltan:
        raise ErrorCompilacion(f"{operacion}: {opt} no generó {', '.join(faltan)}")


def preparar(operaciones, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE, forzar=False, clang=CLANG, opt=OPT):
    """Genera lo que haga falta de los skills 02-03. Devuelve las operaciones compiladas."""
    compiladas = [op for op in operaciones if forzar or desactualizada(op, carpeta, fuente)]
    for operacion in compiladas:
        compilar(operacion, carpeta, fuente, clang, opt)
    return compiladas


def _correr(comando, directorio):
    try:
        proceso = subprocess.run(comando, cwd=directorio, capture_output=True, text=True)
    except FileNotFoundError:
        raise ErrorCompilacion(f"No se encontró {comando[0]}: instale {CLANG} y {OPT} (skills 02-03).") from None
    if proceso.returncode != 0:
        raise ErrorCompilacion(f"Falló «{' '.join(map(str, comando))}» en {directorio}:\n"
                               f"{(proceso.stderr or proceso.stdout).strip()}")
    return proceso.stdout
