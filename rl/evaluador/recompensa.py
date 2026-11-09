"""E07 · Recompensa.

Correcta:              r_t = C_best,t-1 − C_t   (C_best,0 = C_ref)
Penalizada:            r_t = −P_fallo           (accion_invalida, fallo_evaluador,
                                                 fallo_simulador, incorrecta)
error_infraestructura: r_t = None               (no atribuible: se reintenta o excluye)
"""

from .. import estados

FORMULA = "r = C_best_prev - C (correcta); r = -P_fallo (penalizada); None (error_infraestructura)"


def recompensa(estado, costo, mejor_costo_anterior, p_fallo):
    """Devuelve ``(r, detalle)``."""
    detalle = {"formula": FORMULA, "C_best_prev": mejor_costo_anterior, "P_fallo": p_fallo}
    if estado in estados.NO_ATRIBUIBLES:
        return None, detalle
    if estado in estados.PENALIZADOS:
        return -p_fallo, detalle
    if estado != estados.CORRECTA:
        raise ValueError(f"estado desconocido: {estado}")
    if mejor_costo_anterior is None:
        raise ValueError("no hay C_best previo: la referencia debe evaluarse correctamente antes de optimizar")
    return mejor_costo_anterior - costo, detalle


def costo_reportado(estado, costo_medido, p_fallo):
    """Costo que se registra: el medido si es correcta, P_fallo si es penalizada."""
    if estado == estados.CORRECTA:
        return costo_medido
    if estado in estados.PENALIZADOS:
        return p_fallo
    return None


def verificar_penalizacion(p_fallo, costos_correctos):
    """Garantía empírica: P_fallo debe superar todos los costos correctos observados."""
    maximo = max(costos_correctos, default=None)
    if maximo is not None and p_fallo <= maximo:
        return [f"P_fallo = {p_fallo} no supera el mayor costo correcto observado ({maximo})"]
    return []
