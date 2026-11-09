"""E04 · Ejecución del simulador existente (``cgra/src/run_cgra.py``).

Se ejecuta en un proceso separado, con tiempo máximo, apuntando solo a las
copias de la carpeta de evaluación. No se modifica el simulador: los ciclos y
los NOP se leen del ``reporte_ciclos.txt`` que él mismo escribe, y la memoria
final de ``memoria.bin`` (``--write-back`` sobre la copia de trabajo).
"""

import re
import subprocess
import sys
import time
from pathlib import Path

from ..comun import RUN_CGRA, SRC_CGRA

PATRON_TOTAL = re.compile(r"^Total de ciclos:\s+(\d+)", re.M)
# Fila de la tabla "Instrucciones por PE" (sección 5 del reporte).
PATRON_FILA_PE = re.compile(r"^\s+(PE\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+%\s*$", re.M)


def leer_reporte(texto):
    """``(ciclos, nops, nops_por_pe)`` del reporte de ciclos; ``None`` si no aparece."""
    total = PATRON_TOTAL.search(texto)
    filas = PATRON_FILA_PE.findall(texto)
    nops_por_pe = {fila[0]: int(fila[6]) for fila in filas}
    return (int(total.group(1)) if total else None,
            sum(nops_por_pe.values()) if nops_por_pe else None,
            nops_por_pe or None)


def _clasificar(salida):
    """Tipo de fallo a partir del texto que deja el simulador."""
    if "FIFO vacía" in salida or "no tiene vecino" in salida:
        return "bloqueo_comunicacion"
    if "sintaxis" in salida or "no puede ejecutar" in salida or "Operación desconocida" in salida:
        return "instruccion_invalida"
    return "error_inesperado"


def ejecutar(carpeta_eval, t_max_s):
    """Corre el simulador sobre la carpeta aislada y devuelve el resultado de E04."""
    carpeta_eval = Path(carpeta_eval).resolve()  # run_cgra.py corre con cwd=cgra/src
    reporte = carpeta_eval / "reporte_ciclos.txt"
    comando = [
        sys.executable, str(RUN_CGRA), str(carpeta_eval),
        "--cgra", str(carpeta_eval / "cgra.json"),
        "--instrucciones", str(carpeta_eval / "pe_instructions"),
        "--memoria", str(carpeta_eval / "memoria.bin"),
        "--reporte-ciclos", str(reporte),
        "--write-back",
    ]
    inicio = time.monotonic()
    try:
        proceso = subprocess.run(comando, cwd=SRC_CGRA, capture_output=True, text=True, timeout=t_max_s)
    except subprocess.TimeoutExpired as expirado:
        _guardar(carpeta_eval, expirado.stdout, expirado.stderr)
        return {"ok": False, "estado": "tiempo_excedido", "ciclos": None, "nops": None,
                "errores": [{"skill": "E04", "regla": "tiempo_excedido",
                             "detalle": f"el simulador superó t_max = {t_max_s} s"}],
                "t_s": round(time.monotonic() - inicio, 3), "comando": comando}
    t_s = round(time.monotonic() - inicio, 3)
    _guardar(carpeta_eval, proceso.stdout, proceso.stderr)

    if proceso.returncode != 0:
        salida = (proceso.stderr or "") + (proceso.stdout or "")
        ultima = next((l for l in reversed(salida.strip().splitlines()) if l.strip()), "sin mensaje")
        return {"ok": False, "estado": _clasificar(salida), "ciclos": None, "nops": None,
                "codigo_salida": proceso.returncode,
                "errores": [{"skill": "E04", "regla": _clasificar(salida), "detalle": ultima.strip()}],
                "t_s": t_s, "comando": comando}
    if not reporte.is_file() or not (carpeta_eval / "memoria.bin").is_file():
        return {"ok": False, "estado": "sin_salidas", "ciclos": None, "nops": None,
                "errores": [{"skill": "E04", "regla": "sin_salidas",
                             "detalle": "el simulador terminó sin reporte o sin memoria final"}],
                "t_s": t_s, "comando": comando}

    ciclos, nops, nops_por_pe = leer_reporte(reporte.read_text(encoding="utf-8"))
    avisos = [] if nops is not None else ["el reporte no trae la tabla de NOP por PE: nops = null"]
    return {"ok": ciclos is not None, "estado": "termino", "ciclos": ciclos, "nops": nops,
            "nops_por_pe": nops_por_pe, "ruta_memoria": str(carpeta_eval / "memoria.bin"),
            "ruta_reporte": str(reporte), "avisos": avisos,
            "errores": [] if ciclos is not None else [{"skill": "E04", "regla": "reporte_ilegible",
                                                       "detalle": "no se encontró 'Total de ciclos' en el reporte"}],
            "t_s": t_s, "comando": comando}


def _guardar(carpeta, stdout, stderr):
    def texto(valor):
        return valor.decode("utf-8", "replace") if isinstance(valor, bytes) else (valor or "")
    (Path(carpeta) / "simulador_stdout.txt").write_text(texto(stdout), encoding="utf-8")
    (Path(carpeta) / "simulador_stderr.txt").write_text(texto(stderr), encoding="utf-8")
