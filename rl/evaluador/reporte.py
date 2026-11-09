"""E08 · Reporte estructurado: única salida del evaluador hacia el agente.

Formato: sección 4 de ``docs/formatos.md``.
Lo no medido va en ``null``; nunca se estima.
"""

from .. import estados
from .recompensa import FORMULA

CAMPOS_METRICAS = ("C_A", "C_L", "C_R", "C_A_norm", "C_L_norm", "ciclos", "nops", "fraccion_nops")


def parametros_usados(corrida, normalizadores):
    return {
        "pesos_area": corrida["pesos_area"],
        "K": corrida["K"],
        "formula_latencia": corrida["formula_latencia"],
        "tolerancias": corrida["tolerancias"],
        "penalizacion": corrida["penalizacion"],
        "tolerancia_empate_costo": corrida["tolerancia_empate_costo"],
        "normalizadores": normalizadores,
        "formula_recompensa": FORMULA,
        "formula_costo": "C = K_A*C_A/C_A_ref + K_L*C_L/C_L_ref + K_R*C_R",
    }


def construir(id_evaluacion, accion, estado, metricas=None, costo_total=None, recompensa=None,
              validacion_funcional=None, errores=(), tiempo_s=None, parametros=None, detalle_recompensa=None,
              avisos=()):
    if estado not in estados.TODOS:
        raise ValueError(f"estado de evaluación desconocido: {estado}")
    metricas = dict(metricas or {})
    completas = {campo: metricas.get(campo) for campo in CAMPOS_METRICAS}
    for extra in ("capacidades", "nops_por_pe"):
        if metricas.get(extra) is not None:
            completas[extra] = metricas[extra]
    return {
        "id_evaluacion": id_evaluacion,
        "accion": accion,
        "estado_evaluacion": estado,
        "metricas": completas,
        "costo_total": costo_total,
        "costo_penalizado": estado in estados.PENALIZADOS,
        "recompensa": recompensa,
        "detalle_recompensa": detalle_recompensa,
        "validacion_funcional": validacion_funcional,
        "errores": list(errores),
        "avisos": list(avisos),
        "tiempo_evaluacion_s": tiempo_s,
        "parametros": parametros,
    }


def comprobar(reporte):
    """Coherencia mínima del reporte (la misma que exige A04). Devuelve errores."""
    problemas = []
    estado = reporte.get("estado_evaluacion")
    accion = reporte.get("accion") or {}
    if not accion.get("hashes") and estado not in (estados.ACCION_INVALIDA, estados.ERROR_INFRAESTRUCTURA):
        problemas.append("el reporte no identifica el cgra.json evaluado")
    if accion.get("id_candidata") != reporte.get("id_evaluacion"):
        problemas.append("id_evaluacion no coincide con la candidata")
    m = reporte.get("metricas", {})
    if estado == estados.CORRECTA:
        faltan = [c for c in ("C_A", "C_L", "C_R", "C_A_norm", "C_L_norm", "ciclos") if m.get(c) is None]
        if faltan or reporte.get("costo_total") is None or reporte.get("recompensa") is None:
            problemas.append(f"una evaluación correcta debe tener todas sus métricas (faltan {faltan})")
        if not reporte.get("programas_generados"):
            problemas.append("una evaluación correcta debe identificar los PE*.txt generados por los skills")
    elif estado in (estados.ACCION_INVALIDA, estados.FALLO_EVALUADOR):
        if m.get("ciclos") is not None or m.get("C_L") is not None:
            problemas.append(f"{estado} no se simula: no puede traer ciclos")
    if estado != estados.CORRECTA and estado != estados.INCORRECTA and not reporte.get("errores"):
        problemas.append(f"{estado} debe incluir diagnósticos")
    return problemas
