"""Tabla de resultados de todas las configuraciones evaluadas en una corrida.

Para cada configuración (la malla de referencia y cada malla que simuló la exploración) junta todo lo
que midió el evaluador: costos (C, C_A, C_L, C_R y sus normalizados), y por operación los ciclos
(total, de cómputo, de comunicación e inactivos), las instrucciones (por categoría y por instrucción),
los NOPs, las lecturas y escrituras de memoria, las transferencias entre PEs, la latencia, el error y la
correctitud. Marca las configuraciones eficientes (frente de Pareto área-latencia) y, en el modo
``bloques``, en qué orden, en qué episodio y por quién se propuso cada malla y si se aceptó.

Escribe:

    configuraciones.csv              una fila por configuración (columnas por operación: <op>_<dato>)
    configuraciones_operaciones.csv  una fila por configuración y operación, con las instrucciones por tipo
    configuraciones.json             todo lo anterior, con las métricas completas
"""

import csv
import json
from pathlib import Path

DATOS_OPERACION = ("estado", "ciclos", "ciclos_computo", "ciclos_comunicacion", "ciclos_inactivos", "instrucciones",
                   "instrucciones_memoria", "instrucciones_calculo", "instrucciones_comunicacion",
                   "instrucciones_otras", "nops", "fraccion_nops", "lecturas", "escrituras", "transferencias",
                   "C_L", "C_L_norm", "C_R", "costo", "elementos", "elementos_fallidos", "modo")
DETALLE = ("ciclos_computo", "ciclos_comunicacion", "ciclos_inactivos", "instrucciones_memoria",
           "instrucciones_calculo", "instrucciones_comunicacion", "instrucciones_otras", "lecturas", "escrituras",
           "transferencias")


def frente(puntos, x="C_A", y="C_L_norm"):
    """Los puntos no dominados (minimizando ``x`` e ``y``), ordenados por ``x``."""
    eficientes = [p for p in puntos
                  if not any(q[x] <= p[x] and q[y] <= p[y] and (q[x] < p[x] or q[y] < p[y]) for q in puntos)]
    return sorted(eficientes, key=lambda p: (p[x], p[y]))


def datos_operacion(o):
    """Los datos de una ``EvaluacionOperacion`` (lo del reporte de ciclos queda en ``None`` si la
    evaluación es anterior a ``detalle``; las instrucciones se deducen entonces de ranuras − NOPs)."""
    detalle = o.detalle or {}
    instrucciones = detalle.get("instrucciones")
    if instrucciones is None and o.slots is not None and o.nops is not None:
        instrucciones = o.slots - o.nops
    validacion = o.validacion or {}
    datos = {"estado": o.estado, "ciclos": o.ciclos, "instrucciones": instrucciones, "nops": o.nops,
             "fraccion_nops": o.fraccion_nops, "C_L": o.C_L, "C_L_norm": o.C_L_norm, "C_R": o.C_R, "costo": o.costo,
             "elementos": validacion.get("total_elementos"), "elementos_fallidos": validacion.get("elementos_fallidos"),
             "modo": o.modo}
    datos.update({k: detalle.get(k) for k in DETALLE})
    datos["por_instruccion"] = detalle.get("por_instruccion")
    return datos


def fila(malla, evaluacion, origen, info=None):
    """Una configuración con todos sus datos. ``malla``: tupla de filas de letras."""
    m = evaluacion.metricas or {}
    letras = [t for f in malla for t in f]
    operaciones = {op: datos_operacion(o) for op, o in evaluacion.operaciones.items()}

    def total(clave):
        valores = [d[clave] for d in operaciones.values()]
        return sum(valores) if valores and all(v is not None for v in valores) else None

    registro = {
        "nombre": f"{len(malla)}x{len(malla[0])}", "filas": len(malla), "columnas": len(malla[0]),
        "pes": len(letras), "n_M": letras.count("M"), "n_E": letras.count("E"), "n_V": letras.count("V"),
        "malla": [" ".join(f) for f in malla], "origen": origen, "estado": evaluacion.estado,
        "motivo": evaluacion.motivo, "C": evaluacion.costo, "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
        "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"), "ciclos_total": total("ciclos"),
        "instrucciones_total": total("instrucciones"), "nops_total": total("nops"),
        "ciclos": {op: d["ciclos"] for op, d in operaciones.items()}, "operaciones": operaciones,
    }
    registro.update({"orden": None, "episodio": None, "propuesta_por": None, "aceptada": None})
    registro.update(info or {})
    return registro


def info_exploracion(episodios):
    """``{malla: {orden, episodio, propuesta_por, aceptada}}`` de las trayectorias del modo bloques."""
    info, orden = {}, 0
    for ep in episodios or []:
        for t in ep.get("trayectoria", []):
            if t.get("malla") is None:
                continue
            clave = tuple(t["malla"])
            if clave not in info:
                info[clave] = {"orden": orden, "episodio": ep["n"], "propuesta_por": t["quien"], "aceptada": False}
                orden += 1
            info[clave]["aceptada"] = info[clave]["aceptada"] or bool(t.get("aceptada"))
    return info


def marcar_frente(registros):
    correctos = [r for r in registros if r["estado"] == "correcta" and r["C_A"] is not None and r["C_L_norm"] is not None]
    eficientes = {id(r) for r in frente(correctos)}
    for r in registros:
        r["eficiente"] = id(r) in eficientes
    return registros


def construir(evaluaciones, episodios=None):
    """Registros de ``evaluaciones`` = ``[(malla, Evaluacion, origen)]`` (sin repetir mallas), con la
    información de la exploración y el frente de Pareto marcado; ordenados por área."""
    info = info_exploracion(episodios)
    registros, vistas = [], set()
    for malla, evaluacion, origen in evaluaciones:
        clave = tuple(" ".join(f) for f in malla)
        if clave in vistas:
            continue
        vistas.add(clave)
        registros.append(fila(malla, evaluacion, origen, info.get(clave)))
    _distinguir_nombres(registros)
    marcar_frente(registros)
    return sorted(registros, key=lambda r: (r["C_A"] is None, r["C_A"] or 0, r["nombre"]))


MAX_PES_PATRON = 12  # hasta cuántos PEs interiores el nombre lleva la distribución completa de tipos


def patron(malla):
    """La distribución de tipos del interior (sin las columnas de memoria de los bordes), una fila por
    tramo separadas por ``/``: ``("M V E M", "M V V M")`` -> ``"VE/VV"``. ``None`` si el interior tiene más
    de ``MAX_PES_PATRON`` PEs (el nombre sería ilegible)."""
    filas = [f.split() if isinstance(f, str) else list(f) for f in malla]
    interior = [f[1:-1] for f in filas]
    if sum(len(f) for f in interior) > MAX_PES_PATRON:
        return None
    return "/".join("".join(f) for f in interior)


def nombre_malla(malla):
    """``"2x4 [VE/VV]"`` para mallas pequeñas; ``"16x16"`` para las grandes."""
    filas = [f.split() if isinstance(f, str) else list(f) for f in malla]
    base = f"{len(filas)}x{len(filas[0])}"
    p = patron(filas)
    return f"{base} [{p}]" if p else base


def _distinguir_nombres(registros):
    """El nombre de cada configuración muestra dónde quedó cada tipo de PE: en las mallas pequeñas, la
    distribución del interior fila por fila (``2x4 [VE/VV]``: fila 0 = V E, fila 1 = V V, entre las
    columnas M). Si dos mallas grandes distintas tienen el mismo tamaño, llevan sus conteos de E y V y,
    si aún coinciden, un número."""
    for r in registros:
        r["nombre"] = nombre_malla(r["malla"])
        r["distribucion"] = patron(r["malla"])
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


def escribir(salida, registros, prefijo="configuraciones"):
    """Escribe ``<prefijo>.csv``, ``<prefijo>_operaciones.csv`` y ``<prefijo>.json`` en ``salida``."""
    salida = Path(salida)
    operaciones = sorted({op for r in registros for op in r["operaciones"]})
    base = ["nombre", "filas", "columnas", "pes", "n_M", "n_E", "n_V", "distribucion", "malla", "origen", "orden", "episodio",
            "propuesta_por", "aceptada", "estado", "eficiente", "C", "C_A", "C_A_norm", "C_L_norm", "C_R",
            "ciclos_total", "instrucciones_total", "nops_total"]
    with open(salida / f"{prefijo}.csv", "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["configuracion" if c == "nombre" else c for c in base]
                   + [f"{op}_{d}" for op in operaciones for d in DATOS_OPERACION])
        for r in registros:
            w.writerow([" / ".join(r["malla"]) if c == "malla" else r.get(c) for c in base]
                       + [r["operaciones"].get(op, {}).get(d) for op in operaciones for d in DATOS_OPERACION])
    instrucciones = sorted({i for r in registros for d in r["operaciones"].values()
                            for i in (d.get("por_instruccion") or {})})
    with open(salida / f"{prefijo}_operaciones.csv", "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["configuracion", "pes", "C", "eficiente", "operacion"] + list(DATOS_OPERACION)
                   + [f"inst_{i}" for i in instrucciones])
        for r in registros:
            for op in operaciones:
                d = r["operaciones"].get(op)
                if d is None:
                    continue
                por = d.get("por_instruccion") or {}
                w.writerow([r["nombre"], r["pes"], r["C"], r["eficiente"], op] + [d.get(k) for k in DATOS_OPERACION]
                           + [por.get(i, 0 if por else None) for i in instrucciones])
    temporal = salida / f"{prefijo}.json.tmp"
    temporal.write_text(json.dumps({"objetivos_pareto": ["C_A", "C_L_norm"], "configuraciones": registros},
                                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporal.replace(salida / f"{prefijo}.json")
    return registros


def de_corrida(carpeta):
    """``[(malla, Evaluacion, origen)]`` y los episodios de una carpeta de resultados: la malla de
    referencia (``evaluacion_malla.json``) y la caché de la exploración (``evaluaciones_episodios.json``)."""
    from .evaluador import Evaluacion, EvaluacionOperacion
    carpeta = Path(carpeta)

    def evaluacion(d):
        d = dict(d)
        d["operaciones"] = {op: EvaluacionOperacion(**o) for op, o in d["operaciones"].items()}
        return Evaluacion(**d)

    evaluaciones = []
    ruta = carpeta / "evaluacion_malla.json"
    if ruta.exists():
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        evaluaciones.append((tuple(tuple(f.split()) for f in datos["configuracion"]["malla"]),
                             evaluacion(datos["evaluacion"]), "referencia"))
    ruta = carpeta / "evaluaciones_episodios.json"
    if ruta.exists():
        for item in json.loads(ruta.read_text(encoding="utf-8"))["episodios"]:
            evaluaciones.append((tuple(tuple(f.split()) for f in item["configuracion"]["malla"]),
                                 evaluacion(item["evaluacion"]), "exploracion"))
    ruta = carpeta / "episodios.json"
    episodios = json.loads(ruta.read_text(encoding="utf-8"))["episodios"] if ruta.exists() else None
    return evaluaciones, episodios
