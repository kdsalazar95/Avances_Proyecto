"""Datos para comparar los dos sistemas: convergencia, epsilon y optimización de la función de costo.

Es el mismo formato que escribe el Sistema 1 (``rl/herramientas/convergencia.py`` en la rama
``feature/sistema1-llm``), con ``sistema = sistema2_greedy``, para poder juntar los archivos de los dos.

Por corrida (se escriben al terminar ``agente_greedy.py`` o con ``python3 herramientas/convergencia.py DIR``):

    convergencia.csv   una fila por malla propuesta, en orden (n = 0 es la referencia 32x32): quién la propuso
                       (``greedy`` o ``exploracion``), estado, tamaño, C y sus términos (C_A_norm, C_L_norm, C_R),
                       el mejor C hasta ahí (C_best), si mejoró, y los tiempos (decisión, evaluación y acumulado)
    convergencia.json  el resumen: epsilon, motivo de fin, C de la referencia y mejor C, mejora relativa,
                       cuándo convergió (evaluación y tiempo en que se alcanzó el mejor C) y cuándo se alcanzó
                       el 50 %, 90 % y 100 % de la mejora total

Varias corridas, p. ej. una por epsilon (``python3 herramientas/convergencia.py DIR1 DIR2 ... --salida DIR``):

    comparacion_epsilon.csv  una fila por corrida y paciencia con el resumen (la tabla de la comparación)
    curvas_epsilon.csv       C y C_best de cada evaluación de cada corrida y paciencia, en formato largo: las curvas

Paciencia: una corrida con paciencia P contiene las de paciencia menor. La paciencia solo decide cuándo se deja
de intentar: con la misma semilla, los primeros episodios son los mismos y una corrida con paciencia 3 se habría
detenido en el primer punto en que hubo 3 episodios seguidos sin mejorar el mejor C. ``corte`` aplica esa regla
(la misma de ``Agente.correr``) a los episodios guardados, así que de una corrida con paciencia 10 salen también
los resultados de paciencia 3 y 6 (``PACIENCIAS``), sin correrlas aparte.

Definiciones (las mismas del Sistema 1):

* Una fila por malla que el agente propone y evalúa (todas las de los descensos y de la fase de tipos de todos
  los episodios, en orden). El punto de partida de cada episodio (la referencia) no se repite.
* ``tiempo_acumulado_s``: suma de lo que tardaron la decisión (Greedy o el azar) y la evaluación (Dijkstra +
  simulador) desde la primera propuesta. Una malla que ya se había probado en esta corrida no se vuelve a
  evaluar (tiempo de evaluación 0). Una malla que vino de la caché de otra corrida cuenta lo que tardó cuando se
  simuló, así que compartir la caché entre corridas no cambia los tiempos. La referencia es n = 0, tiempo 0.
* ``C_best``: el menor C de las evaluaciones correctas hasta esa fila (empieza en el C de la referencia).
* Convergencia: la primera evaluación (y su tiempo acumulado) en que C_best llega a su valor final.
* Hito p %: la primera evaluación en que C_best ya bajó el p % de la mejora total (C_ref − C_best final).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

CORRECTA = "correcta"
SISTEMA = "sistema2_greedy"
HITOS = (50, 90, 100)
PACIENCIAS = (3, 6, 10)  # las que se calculan cortando cada corrida (solo las <= su propia paciencia)
NO_ATRIBUIBLES = {"error_infraestructura"}  # no cuentan para la paciencia (como en Agente.correr)
COLUMNAS = ("sistema", "corrida", "epsilon", "semilla", "n", "id", "decidio", "estado", "filas", "columnas", "pes",
            "C", "C_A", "C_A_norm", "C_L_norm", "C_R", "C_best", "mejora", "recompensa",
            "tiempo_decision_s", "tiempo_evaluacion_s", "tiempo_acumulado_s", "malla")
COLUMNAS_COMPARACION = ("sistema", "corrida", "epsilon", "paciencia", "semilla", "motivo_fin", "episodios",
                        "evaluaciones",
                        "evaluaciones_correctas", "fraccion_exploracion", "C_ref", "C_best", "mejora_relativa",
                        "evaluacion_convergencia", "tiempo_convergencia_s", "evaluacion_50", "tiempo_50_s",
                        "evaluacion_90", "tiempo_90_s", "tiempo_total_s", "mejor_malla")
COLUMNAS_CURVAS = ("sistema", "corrida", "epsilon", "paciencia", "n", "decidio", "tiempo_acumulado_s", "C", "C_best")
QUIEN = {"explotacion": "greedy", "exploracion": "exploracion"}


def _leer(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _escribir_json(ruta, datos):
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metadatos(carpeta):
    carpeta = Path(carpeta)
    datos = _leer(carpeta / "episodios.json")
    episodios = datos.get("episodios", [])
    return {"corrida": carpeta.name, "epsilon": datos.get("epsilon"),
            "semilla": episodios[0]["semilla"] if episodios else None,
            "limites": {"max_iteraciones": datos.get("max_episodios"), "paciencia": datos.get("paciencia")},
            "motivo_fin": datos.get("motivo_fin")}, datos


def corte(episodios, paciencia, motivo_fin=None):
    """``(n_episodios, motivo)``: cuántos episodios habría hecho la corrida con esta ``paciencia`` (la regla de
    ``Agente.correr``: un episodio que baja el mejor C reinicia la cuenta; uno que no, suma uno, salvo un fallo de
    infraestructura) y por qué habría terminado. Si no se llega a la paciencia, es la corrida entera."""
    sin_mejora = 0
    for i, ep in enumerate(episodios):
        if ep.get("mejora"):
            sin_mejora = 0
        elif ep.get("estado") not in NO_ATRIBUIBLES:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            return i + 1, "paciencia"
    return len(episodios), motivo_fin


def filas_corrida(carpeta, n_episodios=None):
    """Las filas de ``convergencia.csv`` de una corrida del modo bloques (ver el docstring del módulo). Con
    ``n_episodios``, solo las de los primeros episodios (el corte por paciencia)."""
    carpeta = Path(carpeta)
    meta, datos = _metadatos(carpeta)
    ref = _leer(carpeta / "evaluacion_malla.json")["evaluacion"]
    m = ref.get("metricas") or {}
    malla_ref = datos["malla_referencia"]
    c_best = ref["costo"] if ref["estado"] == CORRECTA else None
    base = {"sistema": SISTEMA, "corrida": meta["corrida"], "epsilon": meta["epsilon"], "semilla": meta["semilla"]}
    filas = [{**base, "n": 0, "id": "referencia", "decidio": "referencia", "estado": ref["estado"],
              "filas": len(malla_ref), "columnas": len(malla_ref[0].split()),
              "pes": len(malla_ref) * len(malla_ref[0].split()), "C": ref["costo"], "C_A": m.get("C_A"),
              "C_A_norm": m.get("C_A_norm"), "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"), "C_best": c_best,
              "mejora": False, "recompensa": 0.0, "tiempo_decision_s": 0.0,
              "tiempo_evaluacion_s": ref.get("tiempo_s"), "tiempo_acumulado_s": 0.0, "malla": None}]
    acumulado = 0.0
    for ep in datos["episodios"][:n_episodios]:
        for k, t in enumerate(ep.get("trayectoria", [])):
            if t.get("accion") == "inicio" or t.get("estado") is None:
                continue
            n = len(filas)
            correcta = t["estado"] == CORRECTA and t.get("C") is not None
            recompensa = (c_best - t["C"]) if correcta and c_best is not None else \
                (-t["C"] if t.get("C") is not None and not correcta else None)
            mejora = correcta and (c_best is None or t["C"] < c_best)
            if mejora:
                c_best = t["C"]
            t_decision = t.get("tiempo_decision_s") or 0.0
            t_evaluacion = 0.0 if t.get("repetida") else t.get("tiempo_evaluacion_s")
            acumulado += t_decision + (t_evaluacion or 0.0)
            pes = t["filas"] * t["columnas"]
            filas.append({**base, "n": n, "id": f"ep{ep['n']}-{k}", "decidio": QUIEN.get(t.get("quien"), t.get("quien")),
                          "estado": t["estado"], "filas": t["filas"], "columnas": t["columnas"], "pes": pes,
                          "C": t.get("C"), "C_A": t.get("C_A"), "C_A_norm": t.get("C_A_norm"),
                          "C_L_norm": t.get("C_L_norm"), "C_R": t.get("C_R"), "C_best": c_best, "mejora": bool(mejora),
                          "recompensa": recompensa, "tiempo_decision_s": round(t_decision, 4),
                          "tiempo_evaluacion_s": t_evaluacion, "tiempo_acumulado_s": round(acumulado, 3),
                          "malla": " / ".join(t["malla"]) if pes <= 64 and t.get("malla") else None})
    return filas


def resumen(filas, carpeta, paciencia=None, n_episodios=None, motivo_fin=None):
    """El resumen de convergencia de una corrida a partir de sus filas (mismo formato que el Sistema 1). Con
    ``paciencia``, el de la corrida cortada a esa paciencia (``n_episodios`` y ``motivo_fin`` del corte)."""
    meta, datos = _metadatos(carpeta)
    if paciencia is not None:
        meta = {**meta, "limites": {**meta["limites"], "paciencia": paciencia}, "motivo_fin": motivo_fin}
    agente = [f for f in filas if f["decidio"] != "referencia"]
    ref = filas[0] if filas and filas[0]["decidio"] == "referencia" else None
    c_ref = ref["C"] if ref and ref["estado"] == CORRECTA else None
    c_best = filas[-1]["C_best"] if filas else None
    mejora_total = (c_ref - c_best) if c_ref is not None and c_best is not None else None
    mejor = next((f for f in filas if f["C_best"] == c_best and f["C"] == c_best), None)

    def hito(p):
        if not mejora_total or mejora_total <= 0:
            return None
        meta_c = c_ref - p / 100 * mejora_total
        f = next((f for f in filas if f["C_best"] is not None and f["C_best"] <= meta_c + 1e-12), None)
        return {"evaluacion": f["n"], "tiempo_s": f["tiempo_acumulado_s"]} if f else None

    sin_tiempo = sum(f["tiempo_evaluacion_s"] is None for f in agente)
    exploracion = [f for f in agente if f["decidio"] == "exploracion"]
    return {
        "sistema": SISTEMA,
        **meta,
        "paciencia": meta["limites"]["paciencia"],
        "paciencia_corrida": datos.get("paciencia"),
        "episodios": n_episodios if n_episodios is not None else len(datos["episodios"]),
        "evaluaciones": len(agente),
        "evaluaciones_correctas": sum(f["estado"] == CORRECTA for f in agente),
        "fraccion_exploracion": round(len(exploracion) / len(agente), 3) if agente else None,
        "C_ref": c_ref,
        "C_best": c_best,
        "mejora_relativa": round(mejora_total / c_ref, 6) if mejora_total is not None and c_ref else None,
        "mejor": None if mejor is None else {k: mejor[k] for k in ("n", "id", "decidio", "filas", "columnas", "malla")},
        "convergencia": None if mejor is None else {"evaluacion": mejor["n"], "tiempo_s": mejor["tiempo_acumulado_s"]},
        "hitos": {str(p): hito(p) for p in HITOS},
        "tiempo_total_s": agente[-1]["tiempo_acumulado_s"] if agente else 0.0,
        "evaluaciones_sin_tiempo": sin_tiempo,
        "normalizadores": "sistema2_greedy/config/normalizacion.json",
        "nota": "tiempo_acumulado_s = decisión + evaluación desde la primera propuesta (la referencia es n = 0, "
                "tiempo 0; una malla repetida en la corrida no se vuelve a evaluar). Convergencia: primera evaluación "
                "en que C_best llega a su valor final. Hito p: primera evaluación en que C_best ya bajó el p % de "
                "C_ref − C_best. evaluaciones_sin_tiempo > 0: mallas de una caché anterior a que se guardaran los "
                "tiempos (cuentan 0).",
    }


def _csv(ruta, columnas, filas):
    with Path(ruta).open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)


def por_paciencia(carpeta, paciencias=PACIENCIAS):
    """``[(paciencia, filas, resumen)]`` de la corrida cortada a cada paciencia (las que no superan la suya)."""
    meta, datos = _metadatos(carpeta)
    propia = datos.get("paciencia")
    resultado = []
    for p in sorted(set(paciencias) | ({propia} if propia else set())):
        if propia is not None and p > propia:
            continue  # no se puede saber qué habría pasado con más paciencia que la de la corrida
        n, motivo = corte(datos["episodios"], p, datos.get("motivo_fin"))
        filas = filas_corrida(carpeta, n)
        resultado.append((p, filas, resumen(filas, carpeta, p, n, motivo)))
    return resultado


def escribir(carpeta, paciencias=PACIENCIAS):
    """Escribe ``convergencia.csv`` y ``convergencia.json`` en la carpeta de la corrida y devuelve el resumen.
    ``convergencia.json`` trae también ``por_paciencia``: el resumen de la corrida cortada a cada paciencia."""
    carpeta = Path(carpeta)
    filas = filas_corrida(carpeta)
    datos = resumen(filas, carpeta)
    datos["por_paciencia"] = {str(p): r for p, _, r in por_paciencia(carpeta, paciencias)}
    _csv(carpeta / "convergencia.csv", COLUMNAS, filas)
    _escribir_json(carpeta / "convergencia.json", datos)
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


def comparar(carpetas, salida, paciencias=PACIENCIAS):
    """Junta varias corridas (p. ej. una por epsilon) en ``comparacion_epsilon.csv`` y ``curvas_epsilon.csv``,
    con una fila (y una curva) por corrida y paciencia (cada corrida cortada a cada paciencia)."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    resumenes, curvas = [], []
    for carpeta in carpetas:
        for p, filas, datos in por_paciencia(carpeta, paciencias):
            resumenes.append(datos)
            curvas.extend({**f, "paciencia": p} for f in filas)
    resumenes.sort(key=lambda d: (d["epsilon"], d["paciencia"], d["corrida"]))
    curvas.sort(key=lambda f: (f["epsilon"], f["paciencia"], f["corrida"], f["n"]))
    _csv(salida / "comparacion_epsilon.csv", COLUMNAS_COMPARACION, [fila_comparacion(d) for d in resumenes])
    _csv(salida / "curvas_epsilon.csv", COLUMNAS_CURVAS, curvas)
    _escribir_json(salida / "comparacion_epsilon.json", {"corridas": resumenes})
    return resumenes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corridas", nargs="+", type=Path, help="Carpetas de corridas del modo bloques")
    parser.add_argument("--salida", type=Path, help="Con varias corridas: carpeta de la comparación")
    args = parser.parse_args(argv)
    if args.salida:
        resumenes = comparar(args.corridas, args.salida)
        print(f"{len(resumenes)} corridas: {args.salida}/comparacion_epsilon.csv, curvas_epsilon.csv")
    else:
        for carpeta in args.corridas:
            datos = escribir(carpeta)
            print(f"{carpeta}: C_ref = {datos['C_ref']}, C_best = {datos['C_best']}, convergencia "
                  f"{datos['convergencia']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
