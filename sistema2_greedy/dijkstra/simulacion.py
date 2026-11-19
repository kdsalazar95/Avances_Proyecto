"""E04 · El simulador de ``cgra/src/`` (el mismo de la solución RL), en otro proceso.

Igual que ``rl/medicion/simulacion.py``: no se modifica el simulador. Se corre
``cgra/src/run_cgra.py`` sobre una carpeta de evaluación con tiempo máximo, los
ciclos y los NOPs salen del ``reporte_ciclos.txt`` que él escribe y la memoria
final, de ``memoria.bin`` (``--write-back`` sobre la copia de la carpeta).

Carpeta de evaluación de una operación (la arma ``preparar``)::

    <carpeta>/cgra.json            configuración evaluada (tamaño fijo, malla explícita)
    <carpeta>/memoria.bin          copia de cgra/programas/<operacion>/memoria.bin; al terminar, la final
    <carpeta>/pe_instructions/     PE*.txt de Dijkstra
    <carpeta>/reporte_ciclos.txt   lo escribe el simulador
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from memoria import ImagenMemoria

from . import RUN_CGRA, SRC_SIMULADOR

ARCHIVO_CGRA = "cgra.json"
ARCHIVO_MEMORIA = "memoria.bin"
ARCHIVO_REPORTE = "reporte_ciclos.txt"
CARPETA_PE = "pe_instructions"

PATRON_TOTAL = re.compile(r"^Total de ciclos:\s+(\d+)", re.M)
# Fila de la tabla "Instrucciones por PE" del reporte (la misma expresión del RL).
PATRON_FILA_PE = re.compile(r"^\s+(PE\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+%\s*$", re.M)


PATRON_CICLOS = {
    "ciclos_computo": re.compile(r"^Ciclos de cómputo:\s+(\d+)", re.M),
    "ciclos_comunicacion": re.compile(r"^Ciclos de comunicación:\s+(\d+)", re.M),
    "ciclos_inactivos": re.compile(r"^Ciclos inactivos:\s+(\d+)", re.M),
}
PATRON_INSTRUCCION = re.compile(r"\b([A-Z][A-Z0-9]*) x(\d+)")
PATRON_LECTURAS = re.compile(r"\((\d+) lecturas?\)")
PATRON_ESCRITURAS = re.compile(r"\((\d+) escrituras?\)")
PATRON_PASO = re.compile(r"^\s+Ciclo\s+\d+:", re.M)


def _seccion(texto, titulo):
    """El texto desde la línea que empieza con ``titulo`` hasta la siguiente línea en blanco."""
    i = texto.find(titulo)
    if i < 0:
        return ""
    fin = texto.find("\n\n", i)
    return texto[i:fin if fin >= 0 else len(texto)]


def leer_detalle(texto):
    """Todo lo que el reporte de ciclos dice de la corrida, además del total de ciclos y los NOPs:
    ciclos de cómputo, comunicación e inactivos; instrucciones por categoría (memoria, cálculo,
    comunicación, otras) y por instrucción (ADD, VLD, SEND...); lecturas y escrituras de memoria y
    transferencias entre PEs. Los datos que el reporte no trae quedan en ``None``."""
    detalle = {}
    for clave, patron in PATRON_CICLOS.items():
        encontrado = patron.search(texto)
        detalle[clave] = int(encontrado.group(1)) if encontrado else None
    filas = PATRON_FILA_PE.findall(texto)
    if filas:
        for i, clave in enumerate(("memoria", "calculo", "comunicacion", "otras"), start=2):
            detalle[f"instrucciones_{clave}"] = sum(int(f[i]) for f in filas)
        detalle["instrucciones"] = sum(detalle[f"instrucciones_{c}"]
                                       for c in ("memoria", "calculo", "comunicacion", "otras"))
        detalle["nops"] = sum(int(f[6]) for f in filas)
        detalle["pes_por_tipo"] = {}
        for f in filas:
            detalle["pes_por_tipo"][f[1]] = detalle["pes_por_tipo"].get(f[1], 0) + 1
    por_instruccion = {}
    for nombre, n in PATRON_INSTRUCCION.findall(_seccion(texto, "Instrucciones que ejecuta cada PE")):
        por_instruccion[nombre] = por_instruccion.get(nombre, 0) + int(n)
    detalle["por_instruccion"] = dict(sorted(por_instruccion.items())) or None
    accesos = _seccion(texto, "Accesos a memoria")
    detalle["lecturas"] = sum(int(n) for n in PATRON_LECTURAS.findall(accesos)) if accesos else None
    detalle["escrituras"] = sum(int(n) for n in PATRON_ESCRITURAS.findall(accesos)) if accesos else None
    pasos = texto[texto.find("Pasos de comunicación"):] if "Pasos de comunicación" in texto else ""
    detalle["transferencias"] = len(PATRON_PASO.findall(pasos)) if pasos else None
    return detalle


def leer_reporte(texto):
    """``(ciclos, nops, nops_por_pe)`` del reporte de ciclos; ``None`` si no aparece."""
    total = PATRON_TOTAL.search(texto)
    filas = PATRON_FILA_PE.findall(texto)
    nops_por_pe = {fila[0]: int(fila[6]) for fila in filas}
    return (int(total.group(1)) if total else None,
            sum(nops_por_pe.values()) if nops_por_pe else None,
            nops_por_pe or None)


def preparar(carpeta, configuracion, ruta_memoria):
    """Escribe ``cgra.json`` y copia la memoria inicial en ``carpeta``."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / ARCHIVO_CGRA).write_text(json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    shutil.copyfile(ruta_memoria, carpeta / ARCHIVO_MEMORIA)
    return carpeta


def comando(carpeta, dir_pe=None):
    """La llamada al simulador sobre la carpeta de evaluación (como en el RL)."""
    carpeta = Path(carpeta).resolve()
    return [sys.executable, str(RUN_CGRA), str(carpeta), "--cgra", str(carpeta / ARCHIVO_CGRA),
            "--instrucciones", str(Path(dir_pe).resolve() if dir_pe else carpeta / CARPETA_PE),
            "--memoria", str(carpeta / ARCHIVO_MEMORIA), "--reporte-ciclos", str(carpeta / ARCHIVO_REPORTE),
            "--write-back"]


def _clasificar(salida):
    """Tipo de fallo a partir del texto que deja el simulador (rl/medicion/simulacion.py)."""
    if "FIFO vacía" in salida or "no tiene vecino" in salida:
        return "bloqueo_comunicacion"
    if "sintaxis" in salida or "no puede ejecutar" in salida or "Operación desconocida" in salida:
        return "instruccion_invalida"
    return "error_inesperado"


def leer_corrida(carpeta, ruta_memoria_inicial):
    """Los datos de una corrida terminada: configuración, ciclos, NOPs y memoria inicial y final.

    ``ruta_memoria_inicial`` es la de ``programas/<operacion>/memoria.bin`` (la del contrato)."""
    carpeta = Path(carpeta)
    reporte, memoria = carpeta / ARCHIVO_REPORTE, carpeta / ARCHIVO_MEMORIA
    if not reporte.is_file() or not memoria.is_file():
        raise ValueError(f"{carpeta}: falta {ARCHIVO_REPORTE} o {ARCHIVO_MEMORIA} (¿se corrió con --write-back?)")
    configuracion = json.loads((carpeta / ARCHIVO_CGRA).read_text(encoding="utf-8"))
    texto = reporte.read_text(encoding="utf-8")
    ciclos, nops, nops_por_pe = leer_reporte(texto)
    if ciclos is None or nops is None:
        raise ValueError(f"{reporte}: no tiene 'Total de ciclos' o la tabla de instrucciones por PE")
    n_pes = len(configuracion["malla"]) * len(configuracion["malla"][0].split())
    inicial, _ = ImagenMemoria.leer(ruta_memoria_inicial).regiones()
    final, _ = ImagenMemoria.leer(memoria).regiones()
    return dict(configuracion=configuracion, ciclos={"total": ciclos}, nops=nops, slots=ciclos * n_pes,
                nops_por_pe=nops_por_pe, inicial=inicial, final=final, detalle=leer_detalle(texto))


def ejecutar(formato, dir_pe, ruta_memoria, carpeta, t_max_s):
    """Corre el simulador con tiempo máximo. Devuelve ``(datos, error)``: los datos de
    ``leer_corrida`` o un error E04."""
    from .evaluador import configuracion_de
    preparar(carpeta, configuracion_de(formato), ruta_memoria)
    try:
        proceso = subprocess.run(comando(carpeta, dir_pe), cwd=SRC_SIMULADOR, capture_output=True, text=True,
                                 timeout=t_max_s)
    except subprocess.TimeoutExpired:
        return None, {"skill": "E04", "regla": "tiempo_excedido",
                      "detalle": f"el simulador superó t_max = {t_max_s} s"}
    if proceso.returncode != 0:
        salida = (proceso.stderr or "") + (proceso.stdout or "")
        ultima = next((l for l in reversed(salida.strip().splitlines()) if l.strip()), "sin mensaje")
        return None, {"skill": "E04", "regla": _clasificar(salida), "detalle": ultima.strip()}
    try:
        return leer_corrida(carpeta, ruta_memoria), None
    except (ValueError, OSError) as error:
        return None, {"skill": "E04", "regla": "sin_salidas", "detalle": str(error)}
