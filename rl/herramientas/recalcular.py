"""Recalcula los costos de una corrida con otra fórmula de latencia o con otros K, sin volver a evaluar.

Usa lo que el simulador ya midió y quedó en ``historial.jsonl`` (ciclos y NOPs de cada operación, área
y malla). No toca el historial: escribe ``recalculo_<formula>.json`` en la corrida y devuelve las filas.

    uv run python -m rl recalcular --corrida resultados/corridas/r04-todas-codex --formula ciclos_x_nops
"""

from pathlib import Path

from .. import estados
from ..comun import escribir_json, leer_json, leer_jsonl
from ..evaluador import metricas


def _n_pes(malla):
    return len(malla) * len(malla[0].split()) if malla else None


def _operaciones(entrada):
    """{op: (ciclos, nops)} de una entrada del historial (conjunto u operación única)."""
    if entrada.get("por_operacion"):
        return {op: (d.get("ciclos"), d.get("nops")) for op, d in entrada["por_operacion"].items()}
    m = entrada["metricas"]
    return {"operacion": (m.get("ciclos"), m.get("nops"))}


def recalcular(carpeta_corrida, formula, K=None):
    carpeta = Path(carpeta_corrida)
    corrida = leer_json(carpeta / "corrida.json")
    K = K or corrida["K"]
    entradas = leer_jsonl(carpeta / "historial.jsonl")
    referencia = next((e for e in entradas if e.get("es_referencia")), None)
    if referencia is None or referencia["estado_evaluacion"] != estados.CORRECTA:
        raise ValueError("la corrida no tiene una referencia correcta")
    pes_ref = _n_pes(referencia["malla"])
    c_l_ref = {op: metricas.costo_latencia(c, n, formula, pes_ref) for op, (c, n) in _operaciones(referencia).items()}
    c_a_ref = referencia["metricas"]["C_A"]

    filas = []
    for e in entradas:
        fila = {"id": e["id_evaluacion"], "malla": e.get("malla"), "estado": e["estado_evaluacion"],
                "C_A": e["metricas"].get("C_A"), "costo_original": e["costo_total"], "costo": None,
                "por_operacion": {}}
        if e["estado_evaluacion"] == estados.CORRECTA:
            pes = _n_pes(e["malla"])
            costos = []
            for op, (ciclos, nops) in _operaciones(e).items():
                c_l = metricas.costo_latencia(ciclos, nops, formula, pes)
                costo = K["K_A"] * fila["C_A"] / c_a_ref + K["K_L"] * c_l / c_l_ref[op]
                costos.append(costo)
                fila["por_operacion"][op] = {"ciclos": ciclos, "nops": nops,
                                             "fraccion_nops": metricas.fraccion_nops(ciclos, nops, pes),
                                             "C_L": c_l, "costo": costo}
            fila["costo"] = sum(costos) / len(costos)
        filas.append(fila)
    correctas = [f for f in filas if f["costo"] is not None]
    mejor = min(correctas, key=lambda f: f["costo"]) if correctas else None
    resultado = {"formula_latencia": formula, "K": K, "C_A_ref": c_a_ref, "C_L_ref": c_l_ref,
                 "mejor": mejor and mejor["id"], "filas": filas,
                 "nota": "Recalculado desde historial.jsonl con los ciclos y NOPs medidos; el historial no cambia."}
    escribir_json(carpeta / f"recalculo_{formula}.json", resultado)
    return resultado
