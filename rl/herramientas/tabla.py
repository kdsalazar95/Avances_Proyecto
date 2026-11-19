"""Tabla de resultados de todas las configuraciones evaluadas en una corrida del Sistema 1.

Las mismas columnas que la tabla del Sistema 2 (``sistema2_greedy/dijkstra/tabla.py``): una fila por configuración
distinta (la referencia y cada malla que se evaluó), con sus costos y, por operación, sus ciclos, NOPs, latencia y
error. Marca las eficientes (frente de Pareto área-latencia) y en qué orden, en qué episodio y por quién (``llm`` o
``exploracion``) se propuso cada una, y si pasó a ser la mejor de su episodio (``aceptada``).

Lo que el Sistema 2 lee del detalle del reporte de ciclos (ciclos de cómputo y comunicación, instrucciones por
categoría, lecturas, escrituras, transferencias) queda vacío: el reporte del Sistema 1 no lo trae. Las instrucciones
se deducen de ranuras − NOPs (ranuras = ciclos x PEs), como hace el Sistema 2 cuando falta el detalle.

Escribe en la carpeta de la corrida:

    configuraciones.csv              una fila por configuración (columnas por operación: <op>_<dato>)
    configuraciones_operaciones.csv  una fila por configuración y operación
    configuraciones.json             todo lo anterior
"""

import csv
import json
from pathlib import Path

from .. import estados
from ..comun import escribir_json, leer_jsonl

DATOS_OPERACION = ("estado", "ciclos", "ciclos_computo", "ciclos_comunicacion", "ciclos_inactivos", "instrucciones",
                   "instrucciones_memoria", "instrucciones_calculo", "instrucciones_comunicacion",
                   "instrucciones_otras", "nops", "fraccion_nops", "lecturas", "escrituras", "transferencias",
                   "C_L", "C_L_norm", "C_R", "costo", "elementos", "elementos_fallidos", "modo")
BASE = ["nombre", "filas", "columnas", "pes", "n_M", "n_E", "n_V", "distribucion", "malla", "origen", "orden",
        "episodio", "propuesta_por", "aceptada", "estado", "eficiente", "C", "C_A", "C_A_norm", "C_L_norm", "C_R",
        "ciclos_total", "instrucciones_total", "nops_total"]
MAX_PES_PATRON = 12  # hasta cuántos PEs interiores el nombre lleva la distribución completa de tipos


def frente(puntos, x="C_A", y="C_L_norm"):
    """Los puntos no dominados (minimizando ``x`` e ``y``), ordenados por ``x``."""
    eficientes = [p for p in puntos
                  if not any(q[x] <= p[x] and q[y] <= p[y] and (q[x] < p[x] or q[y] < p[y]) for q in puntos)]
    return sorted(eficientes, key=lambda p: (p[x], p[y]))


def patron(malla):
    """La distribución de tipos del interior, una fila por tramo: ``("M V E M", "M V V M")`` -> ``"VE/VV"``."""
    interior = [f.split()[1:-1] for f in malla]
    if sum(len(f) for f in interior) > MAX_PES_PATRON:
        return None
    return "/".join("".join(f) for f in interior)


def nombre_malla(malla):
    """``"2x4 [VE/VV]"`` para mallas pequeñas; ``"16x16"`` para las grandes (como en el Sistema 2)."""
    base = f"{len(malla)}x{len(malla[0].split())}"
    p = patron(malla)
    return f"{base} [{p}]" if p else base


def _operacion(d, pes, validacion):
    instrucciones = d["ciclos"] * pes - d["nops"] if d.get("ciclos") is not None and d.get("nops") is not None else None
    validacion = validacion or {}
    return {"estado": d.get("estado"), "ciclos": d.get("ciclos"), "instrucciones": instrucciones, "nops": d.get("nops"),
            "fraccion_nops": d.get("fraccion_nops"), "C_L": d.get("C_L"), "C_L_norm": d.get("C_L_norm"),
            "C_R": d.get("C_R"), "costo": d.get("costo"), "elementos": validacion.get("total_elementos"),
            "elementos_fallidos": validacion.get("elementos_fallidos"), "modo": None}


def construir(carpeta):
    """Los registros de la corrida (una configuración distinta por registro), ordenados por área."""
    carpeta = Path(carpeta)
    entradas = leer_jsonl(carpeta / "historial.jsonl")
    aceptadas = set()
    ruta_ep = carpeta / "episodios.json"
    if ruta_ep.is_file():
        aceptadas = {ep["mejor"] for ep in json.loads(ruta_ep.read_text(encoding="utf-8"))["episodios"]}
    registros, por_clave = [], {}
    for orden, e in enumerate(entradas):
        malla = e.get("malla")
        if not malla or e["estado_evaluacion"] == estados.ACCION_INVALIDA:
            continue
        clave = e.get("clave_hardware") or json.dumps(malla)
        if clave in por_clave:
            por_clave[clave]["aceptada"] = por_clave[clave]["aceptada"] or e["id_evaluacion"] in aceptadas
            continue
        letras = [t for f in malla for t in f.split()]
        pes = len(letras)
        m = e.get("metricas") or {}
        validaciones = ((e.get("validacion_funcional") or {}).get("por_operacion") or {})
        operaciones = {op: _operacion(d, pes, validaciones.get(op)) for op, d in (e.get("por_operacion") or {}).items()}

        def total(campo):
            valores = [d[campo] for d in operaciones.values()]
            return sum(valores) if valores and all(v is not None for v in valores) else None

        r = {"nombre": nombre_malla(malla), "filas": len(malla), "columnas": len(malla[0].split()), "pes": pes,
             "n_M": letras.count("M"), "n_E": letras.count("E"), "n_V": letras.count("V"),
             "distribucion": patron(malla), "malla": malla,
             "origen": "referencia" if e.get("es_referencia") else "exploracion",
             "orden": orden, "episodio": (e.get("episodio") or {}).get("n"),
             "propuesta_por": "referencia" if e.get("es_referencia") else e.get("decidio"),
             "aceptada": e["id_evaluacion"] in aceptadas, "estado": e["estado_evaluacion"],
             "C": e.get("costo_total"), "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
             "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"), "ciclos_total": total("ciclos"),
             "instrucciones_total": total("instrucciones"), "nops_total": total("nops"), "operaciones": operaciones,
             "id": e["id_evaluacion"]}
        por_clave[clave] = r
        registros.append(r)
    _distinguir_nombres(registros)
    correctos = [r for r in registros if r["estado"] == estados.CORRECTA and r["C_A"] is not None
                 and r["C_L_norm"] is not None]
    eficientes = {id(r) for r in frente(correctos)}
    for r in registros:
        r["eficiente"] = id(r) in eficientes
    return sorted(registros, key=lambda r: (r["C_A"] is None, r["C_A"] or 0, r["nombre"]))


def _distinguir_nombres(registros):
    """Dos mallas grandes distintas con el mismo tamaño llevan sus conteos de E y V y, si aún coinciden, un número."""
    por_nombre = {}
    for r in registros:
        por_nombre.setdefault(r["nombre"], []).append(r)
    for nombre, grupo in por_nombre.items():
        if len(grupo) < 2:
            continue
        for r in grupo:
            r["nombre"] = f"{nombre} ({r['n_E']}E {r['n_V']}V)"
        vistos = {}
        for r in grupo:
            vistos.setdefault(r["nombre"], []).append(r)
        for repetidos in vistos.values():
            if len(repetidos) > 1:
                for i, r in enumerate(repetidos, start=1):
                    r["nombre"] = f"{r['nombre']} {i}"


def escribir(carpeta, prefijo="configuraciones"):
    """Escribe ``<prefijo>.csv``, ``<prefijo>_operaciones.csv`` y ``<prefijo>.json``; devuelve los registros."""
    carpeta = Path(carpeta)
    registros = construir(carpeta)
    operaciones = sorted({op for r in registros for op in r["operaciones"]})
    with open(carpeta / f"{prefijo}.csv", "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["configuracion" if c == "nombre" else c for c in BASE]
                   + [f"{op}_{d}" for op in operaciones for d in DATOS_OPERACION])
        for r in registros:
            w.writerow([" / ".join(r["malla"]) if c == "malla" else r.get(c) for c in BASE]
                       + [r["operaciones"].get(op, {}).get(d) for op in operaciones for d in DATOS_OPERACION])
    with open(carpeta / f"{prefijo}_operaciones.csv", "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["configuracion", "pes", "C", "eficiente", "operacion"] + list(DATOS_OPERACION))
        for r in registros:
            for op in operaciones:
                d = r["operaciones"].get(op)
                if d is not None:
                    w.writerow([r["nombre"], r["pes"], r["C"], r["eficiente"], op] + [d.get(k) for k in DATOS_OPERACION])
    escribir_json(carpeta / f"{prefijo}.json", {"objetivos_pareto": ["C_A", "C_L_norm"], "configuraciones": registros})
    return registros
