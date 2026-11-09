"""A07 · Conjunto de soluciones correctas empatadas en el menor costo.

    sin óptimas                 -> [candidata], C_min = C
    C <  C_min - tol            -> reemplazar por [candidata]   (mejora estricta)
    |C - C_min| <= tol          -> agregar si no es duplicada   (empate)
    C >  C_min + tol            -> solo historial
Solo participan las evaluaciones con estado ``correcta``.
"""

from .. import estados


def entrada_optima(reporte):
    return {
        "id_candidata": reporte["accion"]["id_candidata"],
        "hash_accion": reporte["accion"]["hash_accion"],
        "directorio": reporte["accion"]["directorio"],
        "costo_total": reporte["costo_total"],
        "metricas": reporte["metricas"],
        "es_referencia": reporte.get("es_referencia", False),
    }


def actualizar_optimas(optimas, reporte, tolerancia):
    """Devuelve ``(nuevas_optimas, C_min, hubo_mejora)`` sin modificar ``optimas``."""
    actuales = list(optimas)
    c_min = min((o["costo_total"] for o in actuales), default=None)
    if reporte["estado_evaluacion"] != estados.CORRECTA or reporte.get("costo_total") is None:
        return actuales, c_min, False
    costo = reporte["costo_total"]
    candidata = entrada_optima(reporte)
    if c_min is None:
        return [candidata], costo, True
    if costo < c_min - tolerancia:
        return [candidata], costo, True
    if abs(costo - c_min) <= tolerancia:
        if all(o["hash_accion"] != candidata["hash_accion"] for o in actuales):
            actuales.append(candidata)
        return actuales, c_min, False
    return actuales, c_min, False
