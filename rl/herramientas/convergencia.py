"""Datos para comparar los dos sistemas: convergencia, epsilon y optimización de la función de costo.

Por corrida (se escriben al terminar ``ciclo`` o con ``python -m rl convergencia --corrida DIR``):

    convergencia.csv   una fila por evaluación, en orden (n = 0 es la referencia): quién la propuso,
                       estado, tamaño, C y sus términos (C_A_norm, C_L_norm, C_R), el mejor C hasta ahí
                       (C_best), si mejoró, y los tiempos (decisión, evaluación y acumulado)
    convergencia.json  el resumen: epsilon, motivo de fin, C de la referencia y mejor C, mejora relativa,
                       cuándo convergió (evaluación y tiempo en que se alcanzó el mejor C) y cuándo se
                       alcanzó el 50 %, 90 % y 100 % de la mejora total

Varias corridas, p. ej. una por epsilon (``python -m rl comparar DIR1 DIR2 ... --salida DIR``):

    comparacion_epsilon.csv  una fila por corrida con el resumen (la tabla de la comparación)
    curvas_epsilon.csv       C y C_best de cada evaluación de cada corrida, en formato largo
                             (sistema, epsilon, n, tiempo_acumulado_s, C, C_best): las curvas de convergencia

El formato es el mismo en los dos sistemas (columna ``sistema``) para poder juntar sus archivos.

Definiciones:

* ``tiempo_acumulado_s``: suma de lo que tardaron la decisión (el LLM o el azar) y la evaluación (Skills +
  simulador) desde la primera iteración. La referencia es n = 0 y tiempo 0: se evalúa una vez, igual para
  todos los epsilon, y no es parte de la búsqueda.
* ``C_best``: el menor C de las evaluaciones correctas hasta esa fila (empieza en el C de la referencia).
* Convergencia: la primera evaluación (y su tiempo acumulado) en que C_best llega a su valor final.
* Hito p %: la primera evaluación en que C_best ya bajó el p % de la mejora total (C_ref − C_best final).
"""

import csv
import json
from pathlib import Path

from .. import estados
from ..comun import escribir_json, leer_json, leer_jsonl, relativa

SISTEMA = "sistema1_llm"
HITOS = (50, 90, 100)
COLUMNAS = ("sistema", "corrida", "epsilon", "semilla", "n", "id", "decidio", "estado", "filas", "columnas", "pes",
            "C", "C_A", "C_A_norm", "C_L_norm", "C_R", "C_best", "mejora", "recompensa",
            "tiempo_decision_s", "tiempo_evaluacion_s", "tiempo_acumulado_s", "malla")
COLUMNAS_COMPARACION = ("sistema", "corrida", "epsilon", "semilla", "motivo_fin", "evaluaciones",
                        "evaluaciones_correctas", "fraccion_exploracion", "C_ref", "C_best", "mejora_relativa",
                        "evaluacion_convergencia", "tiempo_convergencia_s", "evaluacion_50", "tiempo_50_s",
                        "evaluacion_90", "tiempo_90_s", "tiempo_total_s", "mejor_malla")
COLUMNAS_CURVAS = ("sistema", "corrida", "epsilon", "n", "decidio", "tiempo_acumulado_s", "C", "C_best")


def _tamano(malla):
    if not malla:
        return None, None
    return len(malla), len(malla[0].split())


def filas_corrida(carpeta):
    """Las filas de ``convergencia.csv`` de una corrida (ver el docstring del módulo)."""
    carpeta = Path(carpeta)
    corrida = leer_json(carpeta / "corrida.json")
    tolerancia = corrida.get("tolerancia_empate_costo", 0.0)
    filas, c_best, acumulado = [], None, 0.0
    for n, e in enumerate(leer_jsonl(carpeta / "historial.jsonl")):
        referencia = bool(e.get("es_referencia"))
        correcta = e["estado_evaluacion"] == estados.CORRECTA and e.get("costo_total") is not None
        mejora = correcta and (c_best is None or e["costo_total"] < c_best - tolerancia)
        if mejora:
            c_best = e["costo_total"]
        t_decision = 0.0 if referencia else (e.get("tiempo_decision_s") or 0.0)
        t_evaluacion = e.get("tiempo_evaluacion_s") or 0.0
        if not referencia:
            acumulado += t_decision + t_evaluacion
        m = e.get("metricas") or {}
        f, c = _tamano(e.get("malla"))
        filas.append({
            "sistema": SISTEMA, "corrida": corrida["id"], "epsilon": corrida.get("epsilon", 0.0),
            "semilla": corrida.get("semilla", 0), "n": n, "id": e["id_evaluacion"],
            "decidio": "referencia" if referencia else e.get("decidio"), "estado": e["estado_evaluacion"],
            "filas": f, "columnas": c, "pes": f * c if f else None, "C": e.get("costo_total"),
            "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"), "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"),
            "C_best": c_best, "mejora": bool(mejora and not referencia), "recompensa": e.get("recompensa"),
            "tiempo_decision_s": round(t_decision, 3), "tiempo_evaluacion_s": round(t_evaluacion, 3),
            "tiempo_acumulado_s": 0.0 if referencia else round(acumulado, 3),
            "malla": " / ".join(e.get("malla") or []) if f and f * c <= 64 else None,
        })
    return filas


def resumen(filas, carpeta):
    """El resumen de convergencia de una corrida a partir de sus filas."""
    carpeta = Path(carpeta)
    corrida = leer_json(carpeta / "corrida.json")
    estado = leer_json(carpeta / "resumen.json").get("estado") if (carpeta / "resumen.json").is_file() else None
    agente = [f for f in filas if f["decidio"] != "referencia"]
    ref = next((f for f in filas if f["decidio"] == "referencia"), None)
    c_ref = ref["C"] if ref and ref["estado"] == estados.CORRECTA else None
    c_best = filas[-1]["C_best"] if filas else None
    mejora_total = (c_ref - c_best) if c_ref is not None and c_best is not None else None
    mejor = next((f for f in filas if f["C_best"] == c_best and f["C"] == c_best), None)

    def hito(p):
        if not mejora_total or mejora_total <= 0:
            return None
        meta = c_ref - p / 100 * mejora_total
        f = next((f for f in filas if f["C_best"] is not None and f["C_best"] <= meta + 1e-12), None)
        return {"evaluacion": f["n"], "tiempo_s": f["tiempo_acumulado_s"]} if f else None

    exploracion = [f for f in agente if f["decidio"] == "exploracion"]
    return {
        "sistema": SISTEMA,
        "corrida": corrida["id"],
        "epsilon": corrida.get("epsilon", 0.0),
        "semilla": corrida.get("semilla", 0),
        "limites": {k: corrida["limites"].get(k) for k in ("max_iteraciones", "paciencia")},
        "motivo_fin": estado,
        "evaluaciones": len(agente),
        "evaluaciones_correctas": sum(f["estado"] == estados.CORRECTA for f in agente),
        "fraccion_exploracion": round(len(exploracion) / len(agente), 3) if agente else None,
        "C_ref": c_ref,
        "C_best": c_best,
        "mejora_relativa": round(mejora_total / c_ref, 6) if mejora_total is not None and c_ref else None,
        "mejor": None if mejor is None else {k: mejor[k] for k in ("n", "id", "decidio", "filas", "columnas", "malla")},
        "convergencia": None if mejor is None else {"evaluacion": mejor["n"], "tiempo_s": mejor["tiempo_acumulado_s"]},
        "hitos": {str(p): hito(p) for p in HITOS},
        "tiempo_total_s": agente[-1]["tiempo_acumulado_s"] if agente else 0.0,
        "normalizadores": relativa(carpeta / "normalizacion.json"),
        "nota": "tiempo_acumulado_s = decisión + evaluación desde la primera iteración (la referencia es n = 0, "
                "tiempo 0). Convergencia: primera evaluación en que C_best llega a su valor final. Hito p: primera "
                "evaluación en que C_best ya bajó el p % de C_ref − C_best.",
    }


def _csv(ruta, columnas, filas):
    with Path(ruta).open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)


def escribir(carpeta):
    """Escribe ``convergencia.csv`` y ``convergencia.json`` en la carpeta de la corrida y devuelve el resumen."""
    carpeta = Path(carpeta)
    filas = filas_corrida(carpeta)
    datos = resumen(filas, carpeta)
    _csv(carpeta / "convergencia.csv", COLUMNAS, filas)
    escribir_json(carpeta / "convergencia.json", datos)
    return datos


def fila_comparacion(datos):
    hitos = datos["hitos"]
    return {
        **{k: datos.get(k) for k in COLUMNAS_COMPARACION if k in datos},
        "evaluacion_convergencia": (datos["convergencia"] or {}).get("evaluacion"),
        "tiempo_convergencia_s": (datos["convergencia"] or {}).get("tiempo_s"),
        "evaluacion_50": (hitos.get("50") or {}).get("evaluacion"), "tiempo_50_s": (hitos.get("50") or {}).get("tiempo_s"),
        "evaluacion_90": (hitos.get("90") or {}).get("evaluacion"), "tiempo_90_s": (hitos.get("90") or {}).get("tiempo_s"),
        "mejor_malla": (datos["mejor"] or {}).get("malla") or
                       (f"{datos['mejor']['filas']}x{datos['mejor']['columnas']}" if datos.get("mejor") else None),
    }


def comparar(carpetas, salida):
    """Junta varias corridas (p. ej. una por epsilon) en ``comparacion_epsilon.csv`` y ``curvas_epsilon.csv``."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    resumenes, curvas = [], []
    for carpeta in carpetas:
        filas = filas_corrida(carpeta)
        resumenes.append(resumen(filas, carpeta))
        curvas.extend(filas)
    resumenes.sort(key=lambda d: (d["epsilon"], d["corrida"]))
    curvas.sort(key=lambda f: (f["epsilon"], f["corrida"], f["n"]))
    _csv(salida / "comparacion_epsilon.csv", COLUMNAS_COMPARACION, [fila_comparacion(d) for d in resumenes])
    _csv(salida / "curvas_epsilon.csv", COLUMNAS_CURVAS, curvas)
    escribir_json(salida / "comparacion_epsilon.json", {"corridas": resumenes})
    return resumenes
