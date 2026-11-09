"""Informe de una corrida: configuraciones óptimas y no óptimas (``informe.md`` y ``informe.json``).

Se arma solo desde ``historial.jsonl`` (lo que midieron Codex + el simulador) y se reescribe después de
cada lote. Secciones:

1. Óptima global (menor costo C).
2. Mejor configuración de cada tamaño de malla (filas x columnas).
3. Frente de Pareto: correctas que ninguna otra supera en área **y** latencia a la vez (no dependen de K).
4. Todas las configuraciones, ordenadas por costo, marcadas como óptima / Pareto / no óptima, con el motivo.
5. Uso de cada PE (porcentaje de ciclos sin NOP) en las mejores, para decidir qué PEs reacomodar.
6. Configuraciones que fallaron y por qué.

    uv run python -m rl informe --corrida resultados/corridas/<corrida>
"""

from pathlib import Path

from .. import estados
from ..comun import ahora, escribir_json, leer_json, leer_jsonl


def _tamano(malla):
    return f"{len(malla)}x{len(malla[0].split())}" if malla else "?"


def _clave_tamano(t):
    try:
        f, c = t.split("x")
        return int(f) * int(c), int(f), int(c)
    except ValueError:
        return (10 ** 9, 0, 0)


def _malla(malla):
    return " / ".join(malla or [])


def _ocupacion(entrada):
    """{op: {pe: % de ciclos sin NOP}} de una entrada correcta."""
    salida = {}
    for op, d in (entrada.get("por_operacion") or {"operacion": entrada["metricas"]}).items():
        ciclos, por_pe = d.get("ciclos"), d.get("nops_por_pe")
        if ciclos and por_pe:
            salida[op] = {pe: round(100 * (1 - n / ciclos)) for pe, n in sorted(por_pe.items())}
    return salida


def _pareto(correctas):
    """Correctas no dominadas en (C_A, C_L_norm): ninguna otra es <= en ambas y < en una."""
    frente = []
    for e in correctas:
        a, l = e["metricas"]["C_A"], e["metricas"]["C_L_norm"]
        dominada = any(o is not e and o["metricas"]["C_A"] <= a and o["metricas"]["C_L_norm"] <= l and
                       (o["metricas"]["C_A"] < a or o["metricas"]["C_L_norm"] < l) for o in correctas)
        if not dominada:
            frente.append(e)
    return sorted(frente, key=lambda e: e["metricas"]["C_A"])


def _dominante(e, correctas):
    a, l = e["metricas"]["C_A"], e["metricas"]["C_L_norm"]
    for o in sorted(correctas, key=lambda x: x["costo_total"]):
        if o is not e and o["metricas"]["C_A"] <= a and o["metricas"]["C_L_norm"] <= l and \
                (o["metricas"]["C_A"] < a or o["metricas"]["C_L_norm"] < l):
            return o["id_evaluacion"]
    return None


def _pct(valor):
    return "—" if valor is None else f"{100 * valor:.0f}%"


def _ciclos_op(e):
    po = e.get("por_operacion")
    if po:
        return " · ".join(f"{op[:4]} {d['ciclos']}" for op, d in po.items())
    return str(e["metricas"].get("ciclos"))


def _primer_error(e):
    for x in e.get("errores", []):
        op = f"{x['operacion']}: " if x.get("operacion") else ""
        return f"{op}[{x.get('skill')}/{x.get('regla')}] {str(x.get('detalle'))[:160]}"
    return ""


def escribir_informe(carpeta_corrida):
    carpeta = Path(carpeta_corrida)
    corrida = leer_json(carpeta / "corrida.json")
    entradas = leer_jsonl(carpeta / "historial.jsonl") if (carpeta / "historial.jsonl").is_file() else []
    correctas = [e for e in entradas if e["estado_evaluacion"] == estados.CORRECTA]
    agente = [e for e in correctas if not e.get("es_referencia")]
    referencia = next((e for e in entradas if e.get("es_referencia")), None)
    optima = min(correctas, key=lambda e: e["costo_total"]) if correctas else None
    frente = _pareto(correctas)
    ids_frente = {e["id_evaluacion"] for e in frente}

    por_tamano = {}
    for e in correctas:
        t = _tamano(e.get("malla"))
        if t not in por_tamano or e["costo_total"] < por_tamano[t]["costo_total"]:
            por_tamano[t] = e
    evaluadas_por_tamano = {}
    for e in entradas:
        evaluadas_por_tamano.setdefault(_tamano(e.get("malla")), []).append(e)

    ops = corrida.get("operaciones") or [corrida.get("operacion")]
    L = [f"# Informe de la corrida `{corrida['id']}`", "",
         f"Actualizado: {ahora()} · operaciones: {', '.join(ops)} · latencia: `{corrida['formula_latencia']}` · "
         f"K = {corrida['K']} · pesos de área = {corrida['pesos_area']}", "",
         f"Evaluadas: {len(entradas)} ({len(correctas)} correctas). Costo C = K_A·C_A/C_A,ref + "
         "K_L·C_L/C_L,ref + K_R·C_R (promedio sobre las operaciones); la referencia vale "
         f"{referencia['costo_total'] if referencia else '—'}. Menor es mejor.", ""]

    L += ["## 1. Óptima", ""]
    if optima:
        L += [f"**{optima['id_evaluacion']}** · `{_malla(optima.get('malla'))}` ({_tamano(optima.get('malla'))}) · "
              f"C = **{optima['costo_total']:.4f}** · área {optima['metricas']['C_A']} · ciclos {_ciclos_op(optima)} · "
              f"NOP {_pct(optima['metricas'].get('fraccion_nops'))}", ""]
    else:
        L += ["Todavía no hay configuraciones correctas.", ""]

    L += ["## 2. Mejor de cada tamaño", "",
          "| Tamaño | Evaluadas | Mejor | Malla | C | Área | Ciclos | NOP |", "|---|---|---|---|---|---|---|---|"]
    for t in sorted(evaluadas_por_tamano, key=_clave_tamano):
        e = por_tamano.get(t)
        n = len(evaluadas_por_tamano[t])
        if e:
            L.append(f"| {t} | {n} | {e['id_evaluacion']} | `{_malla(e.get('malla'))}` | {e['costo_total']:.4f} | "
                     f"{e['metricas']['C_A']} | {_ciclos_op(e)} | {_pct(e['metricas'].get('fraccion_nops'))} |")
        else:
            L.append(f"| {t} | {n} | — | ninguna correcta | — | — | — | — |")
    L.append("")

    L += ["## 3. Frente de Pareto (área contra latencia)", "",
          "Ninguna otra configuración correcta tiene a la vez menos área y menos latencia. Según los K, la mejor "
          "es una de estas.", "",
          "| id | Malla | Área | Latencia normalizada | C |", "|---|---|---|---|---|"]
    for e in frente:
        L.append(f"| {e['id_evaluacion']} | `{_malla(e.get('malla'))}` | {e['metricas']['C_A']} | "
                 f"{e['metricas']['C_L_norm']:.3f} | {e['costo_total']:.4f} |")
    L.append("")

    L += ["## 4. Todas las configuraciones", "",
          "| # | id | Tamaño | Malla | Estado | C | Área | Ciclos | NOP | Clase | Motivo |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    orden = sorted(entradas, key=lambda e: (e["estado_evaluacion"] != estados.CORRECTA, e["costo_total"] or 1e9))
    for k, e in enumerate(orden, 1):
        if e["estado_evaluacion"] != estados.CORRECTA:
            clase, motivo = "no óptima", f"{e['estado_evaluacion']}: {_primer_error(e)}"
        elif optima and e["id_evaluacion"] == optima["id_evaluacion"]:
            clase, motivo = "**óptima**", "menor costo"
        elif e["id_evaluacion"] in ids_frente:
            clase, motivo = "Pareto", "no dominada (otro balance área/latencia)"
        else:
            clase, motivo = "no óptima", f"dominada por {_dominante(e, correctas)}"
        costo = "—" if e["costo_total"] is None else f"{e['costo_total']:.4f}"
        nop = e["metricas"].get("fraccion_nops")
        L.append(f"| {k} | {e['id_evaluacion']} | {_tamano(e.get('malla'))} | `{_malla(e.get('malla'))}` | "
                 f"{e['estado_evaluacion']} | {costo} | {e['metricas'].get('C_A') or '—'} | {_ciclos_op(e)} | "
                 f"{'—' if nop is None else f'{100 * nop:.0f}%'} | {clase} | {motivo.replace('|', '/')} |")
    L.append("")

    L += ["## 5. Uso de cada PE en las mejores", "",
          "Porcentaje de ciclos en que cada PE ejecuta algo distinto de NOP. Un PE con poco uso es candidato a "
          "cambiar de tipo o de posición.", ""]
    for e in sorted(agente, key=lambda x: x["costo_total"])[:5]:
        L.append(f"**{e['id_evaluacion']}** `{_malla(e.get('malla'))}`")
        for op, usos in _ocupacion(e).items():
            L.append(f"- {op}: " + ", ".join(f"{pe} {u}%" for pe, u in usos.items()))
        L.append("")

    fallidas = [e for e in entradas if e["estado_evaluacion"] != estados.CORRECTA]
    L += ["## 6. Fallidas", ""]
    L += [f"- {e['id_evaluacion']} `{_malla(e.get('malla'))}`: {e['estado_evaluacion']} — {_primer_error(e)}"
          for e in fallidas] or ["Ninguna."]
    L.append("")

    ruta = carpeta / "informe.md"
    ruta.write_text("\n".join(L), encoding="utf-8")
    escribir_json(carpeta / "informe.json", {
        "actualizado": ahora(), "optima": optima and optima["id_evaluacion"],
        "pareto": sorted(ids_frente),
        "mejor_por_tamano": {t: e["id_evaluacion"] for t, e in por_tamano.items()},
        "no_optimas": [e["id_evaluacion"] for e in entradas
                       if e["id_evaluacion"] not in ids_frente],
    })
    return ruta
