"""Skill 02 desde Python: ``operaciones.ll`` y ``memoria.bin`` de cada operación.

Es lo mismo que hace el skill 02 (``agent_skills/02-generate-llvm-ir.md``),
como etapa independiente, accesible mediante ``dijkstra_greedy/compilar_cgra.py``:

    cd programas/<operacion>
    clang-18 -S -emit-llvm ../../operaciones/operaciones.c -o operaciones.ll
    clang-18 ../../operaciones/operaciones.c -o operaciones.out && ./operaciones.out <operacion>
    rm operaciones.out

Los dos archivos dependen solo de ``operaciones.c`` (y de ``memoria_cgra.h``),
no de la configuración de la CGRA: se generan una vez y sirven para todas las
mallas. Solo se vuelven a generar si faltan o si el código fuente es más nuevo.
"""

import os
import subprocess
from pathlib import Path

from . import CARPETA_PROGRAMAS, FUENTE_OPERACIONES

CLANG = "clang-18"
FUENTE = FUENTE_OPERACIONES
SALIDAS = ("operaciones.ll", "memoria.bin")
MENSAJE_MEMORIA = "Memoria inicial guardada en memoria.bin"


class ErrorCompilacion(RuntimeError):
    """El skill 02 falló: clang, el programa o el memoria.bin que debía escribir."""


def desactualizada(operacion, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE):
    """Verdadero si a la operación le falta un archivo del skill 02 o alguno es
    más viejo que operaciones.c o que los .h de su carpeta."""
    fuente = Path(fuente)
    salidas = [Path(carpeta) / operacion / nombre for nombre in SALIDAS]
    if not all(s.exists() for s in salidas):
        return True
    cambio = max(f.stat().st_mtime for f in [fuente, *fuente.parent.glob("*.h")])
    return min(s.stat().st_mtime for s in salidas) < cambio


def compilar(operacion, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE, clang=CLANG):
    """Corre los comandos del skill 02 en ``carpeta/<operacion>/``."""
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


def preparar(operaciones, carpeta=CARPETA_PROGRAMAS, fuente=FUENTE, forzar=False, clang=CLANG):
    """Genera lo que haga falta del skill 02. Devuelve las operaciones compiladas."""
    compiladas = [op for op in operaciones if forzar or desactualizada(op, carpeta, fuente)]
    for operacion in compiladas:
        compilar(operacion, carpeta, fuente, clang)
    return compiladas


def _correr(comando, directorio):
    try:
        proceso = subprocess.run(comando, cwd=directorio, capture_output=True, text=True)
    except FileNotFoundError:
        raise ErrorCompilacion(f"No se encontró {comando[0]}: instale {CLANG} (skill 02).") from None
    if proceso.returncode != 0:
        raise ErrorCompilacion(f"Falló «{' '.join(map(str, comando))}» en {directorio}:\n"
                               f"{(proceso.stderr or proceso.stdout).strip()}")
    return proceso.stdout
