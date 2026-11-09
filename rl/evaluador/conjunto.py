"""Evaluación de una configuración con **todas** las operaciones de la corrida.

La CGRA funciona como un procesador: no sabe qué operación va a correr. Una configuración
(``cgra.json``) se evalúa con cada operación del conjunto de la corrida (``"operaciones"`` en
``corrida.json``): por cada una, los Skills 01–09 generan sus PE*.txt y el simulador los ejecuta
sin que se le diga cuál es (solo recibe la carpeta con los programas y la memoria).

Agregación (``solucion_rl.md``, "Programas .c parametrizados"):

- ``correcta`` solo si es correcta en **todas** las operaciones; si una falla, la configuración
  toma el estado de esa falla (un error de infraestructura tiene prioridad: no es culpa del agente);
- costo = promedio de los costos de cada operación. El área es la misma para todas y la latencia
  de cada operación se normaliza con la de la referencia **en esa operación**:
  ``C = K_A·C_A/C_A,ref + K_L·promedio_op(C_L,op/C_L,ref,op) + K_R·promedio_op(C_R,op)``;
- la recompensa se calcula una sola vez, sobre ese costo.

Cada operación se evalúa en ``eval/<id>/<operacion>/`` con el mismo ``evaluar`` de una operación,
y las operaciones corren en paralelo (cada una con su propia sesión del LLM).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import estados
from ..comun import escribir_json
from . import recompensa, reporte
from .evaluacion import accion_de, evaluar

# Si varias operaciones fallan, el estado de la configuración es el primero de esta lista que aparezca.
PRIORIDAD_FALLA = (estados.ERROR_INFRAESTRUCTURA, estados.FALLO_EVALUADOR, estados.FALLO_SIMULADOR,
                   estados.INCORRECTA)


def _promedio(valores):
    valores = [v for v in valores if v is not None]
    return sum(valores) / len(valores) if valores else None


def evaluar_conjunto(dir_candidata, id_evaluacion, instancias, corrida, normalizadores, mejor_costo_anterior,
                     carpeta_eval, compilador, paralelo=True):
    """Evalúa ``dir_candidata`` con cada operación de ``instancias`` ({operacion: instancia}).

    ``normalizadores`` es None para la referencia; si no, ``{"C_A_ref", "operaciones": {op: {...}}}``.
    Devuelve un reporte E08 con ``por_operacion`` y escribe ``eval/<id>/reporte.json``.
    """
    inicio = time.monotonic()
    carpeta_eval = Path(carpeta_eval)
    carpeta_eval.mkdir(parents=True, exist_ok=True)
    p_fallo = corrida["penalizacion"]["P_fallo"]
    es_referencia = normalizadores is None
    operaciones = list(instancias)

    def una(op):
        norm_op = None if es_referencia else normalizadores["operaciones"][op]
        # La recompensa de cada operación es solo informativa: la que cuenta se calcula abajo, una vez.
        return op, evaluar(dir_candidata, id_evaluacion, instancias[op], corrida, norm_op, mejor_costo_anterior,
                           carpeta_eval / op, compilador)

    if paralelo and len(operaciones) > 1:
        with ThreadPoolExecutor(max_workers=len(operaciones)) as hilos:
            resultados = dict(hilos.map(una, operaciones))
    else:
        resultados = dict(una(op) for op in operaciones)

    # Estado agregado.
    estados_op = {op: r["estado_evaluacion"] for op, r in resultados.items()}
    estado = estados.CORRECTA
    for candidato in PRIORIDAD_FALLA:
        if candidato in estados_op.values():
            estado = candidato
            break

    # Métricas agregadas.
    m_op = {op: r["metricas"] for op, r in resultados.items()}
    primera = next(iter(m_op.values()))
    ciclos = [m.get("ciclos") for m in m_op.values()]
    nops = [m.get("nops") for m in m_op.values()]
    metricas = {
        "C_A": primera.get("C_A"),
        "C_A_norm": primera.get("C_A_norm"),
        "C_L": sum(ciclos) if None not in ciclos else None,
        "C_L_norm": _promedio([m.get("C_L_norm") for m in m_op.values()]) if None not in ciclos else None,
        "C_R": _promedio([m.get("C_R") for m in m_op.values()]) if None not in ciclos else None,
        "ciclos": sum(ciclos) if None not in ciclos else None,
        "nops": sum(nops) if None not in nops else None,
        "fraccion_nops": _promedio([m.get("fraccion_nops") for m in m_op.values()]),
    }
    if "capacidades" in primera:
        metricas["capacidades"] = primera["capacidades"]

    costo = None
    if estado == estados.CORRECTA:
        costo = _promedio([r["costo_total"] for r in resultados.values()])
    costo_final = recompensa.costo_reportado(estado, costo, p_fallo)
    if es_referencia and estado == estados.CORRECTA:
        r, detalle = 0.0, {"formula": "referencia: C_best_0 = C_ref", "C_best_prev": None, "P_fallo": p_fallo}
    else:
        r, detalle = recompensa.recompensa(estado, costo_final, mejor_costo_anterior, p_fallo)

    # Normalizadores: los de la referencia, por operación.
    normalizadores_ = normalizadores
    if es_referencia and estado == estados.CORRECTA:
        por_op = {op: res["parametros"]["normalizadores"] for op, res in resultados.items()}
        normalizadores_ = {"C_A_ref": next(iter(por_op.values()))["C_A_ref"], "operaciones": por_op,
                           "referencia": id_evaluacion}

    errores = [{**e, "operacion": op} for op, res in resultados.items() for e in res["errores"]]
    avisos = [f"{op}: {a}" for op, res in resultados.items() for a in res.get("avisos", [])]
    validacion = {
        "aprobada": all((res.get("validacion_funcional") or {}).get("aprobada") for res in resultados.values()),
        "por_operacion": {op: res.get("validacion_funcional") for op, res in resultados.items()},
    }
    parametros = reporte.parametros_usados(corrida, normalizadores_)
    parametros["formula_costo"] = ("C = promedio sobre las operaciones de (K_A*C_A/C_A_ref + "
                                   "K_L*C_L,op/C_L_ref,op + K_R*C_R,op)")
    doc = reporte.construir(id_evaluacion, accion_de(id_evaluacion, dir_candidata), estado, metricas, costo_final,
                            r, validacion, errores, round(time.monotonic() - inicio, 3), parametros, detalle, avisos)
    programas = {op: res.get("programas_generados") for op, res in resultados.items()}
    doc["programas_generados"] = programas if all(programas.values()) else None
    doc["por_operacion"] = {
        op: {"estado": res["estado_evaluacion"], "costo": res["costo_total"],
             "ciclos": res["metricas"].get("ciclos"), "nops": res["metricas"].get("nops"),
             "fraccion_nops": res["metricas"].get("fraccion_nops"), "C_L": res["metricas"].get("C_L"),
             "nops_por_pe": res["metricas"].get("nops_por_pe"),
             "C_L_norm": res["metricas"].get("C_L_norm"), "C_R": res["metricas"].get("C_R"),
             "intentos_skills": (res.get("programas_generados") or {}).get("intentos")}
        for op, res in resultados.items()
    }
    doc["es_referencia"] = es_referencia
    escribir_json(carpeta_eval / "reporte.json", doc)
    return doc
