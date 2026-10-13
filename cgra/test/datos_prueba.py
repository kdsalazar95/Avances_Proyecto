"""Programa de prueba: la reducción 2x2 de 12 elementos, escrito como lo
generan los skills (memoria.bin + un PE*.txt por PE).

Cada PE calcula c[i] = a[i] + b[i] para sus 3 elementos (i = pe, pe + 4,
pe + 8) y los suma en acc (ciclos 0-18). Después PE01 -> PE00 y
PE11 -> PE10 (ciclo 19), y PE10 -> PE00 (ciclo 21), que guarda result.
Con a = b = [1, 2, 3, 4] * 3, c = [2, 4, 6, 8] * 3 y result = 60.
"""

import os
import sys

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from memoria import ImagenMemoria

PES = ("PE00", "PE01", "PE10", "PE11")

REGIONES = {"a": [1, 2, 3, 4] * 3, "b": [1, 2, 3, 4] * 3, "c": [0] * 12, "result": [0]}
FORMAS = {"a": (1, 12), "b": (1, 12), "c": (1, 12), "result": (1, 1)}

RUTA_REDUCCION = {
    "PE00": ["RECV east, rT", "ADD acc, acc, rT", "RECV south, rT", "ADD acc, acc, rT", "ST result[0], acc"],
    "PE01": ["SEND west, acc", "NOP", "NOP", "NOP", "NOP"],
    "PE10": ["RECV east, rT", "ADD acc, acc, rT", "SEND north, acc", "NOP", "NOP"],
    "PE11": ["SEND west, acc", "NOP", "NOP", "NOP", "NOP"],
}


def instrucciones_reduccion_2x2(pe_id):
    """Las 24 instrucciones de un PE, en orden de ciclo."""
    indice = PES.index(pe_id)
    elementos = (indice, indice + 4, indice + 8)
    instrucciones = ["MOV acc, 0.0"]
    for i in elementos:
        instrucciones += [f"LD rA, a[{i}]", f"LD rB, b[{i}]", "ADD rC, rA, rB", f"ST c[{i}], rC"]
    for i in elementos:
        instrucciones += [f"LD rC, c[{i}]", "ADD acc, acc, rC"]
    return instrucciones + RUTA_REDUCCION[pe_id]


def escribir_reduccion_2x2(carpeta):
    """Escribe memoria.bin y pe_instructions/ en ``carpeta``; devuelve sus rutas."""
    os.makedirs(carpeta, exist_ok=True)
    ruta_memoria = os.path.join(carpeta, "memoria.bin")
    ruta_instrucciones = os.path.join(carpeta, "pe_instructions")
    ImagenMemoria.desde_regiones(REGIONES, FORMAS).guardar(ruta_memoria)
    os.makedirs(ruta_instrucciones, exist_ok=True)
    for pe_id in PES:
        with open(os.path.join(ruta_instrucciones, f"{pe_id}.txt"), "w", encoding="utf-8") as archivo:
            archivo.write(f"// {pe_id} instruction list\n\n")
            for ciclo, instruccion in enumerate(instrucciones_reduccion_2x2(pe_id)):
                archivo.write(f"{ciclo:02d}: {instruccion}\n")
    return ruta_memoria, ruta_instrucciones
