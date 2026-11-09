"""A04 · Recepción del reporte del evaluador (solo lectura).

Verifica formato, identidad (id y hash del cgra.json enviado) y coherencia,
y lo convierte en (A_t, M_t, C_t, r_t, e_t) para A05. Un reporte incoherente
es un error de integración: no se completa con supuestos.
"""

from .. import estados
from ..comun import ARCHIVO_CONFIG, hashes_paquete
from ..evaluador import reporte as e08

CAMPOS = ("id_evaluacion", "accion", "estado_evaluacion", "metricas", "costo_total", "recompensa",
          "validacion_funcional", "errores", "tiempo_evaluacion_s", "parametros")


class ErrorIntegracion(ValueError):
    """El reporte no corresponde a la acción enviada o es incoherente."""


def recibir_reporte(reporte, id_enviado, dir_candidata=None):
    faltan = [c for c in CAMPOS if c not in reporte]
    if faltan:
        raise ErrorIntegracion(f"reporte sin los campos {', '.join(faltan)}")
    if reporte["id_evaluacion"] != id_enviado:
        raise ErrorIntegracion(f"el reporte es de '{reporte['id_evaluacion']}' y se envió '{id_enviado}'")
    if reporte["estado_evaluacion"] not in estados.TODOS:
        raise ErrorIntegracion(f"estado desconocido: {reporte['estado_evaluacion']}")
    problemas = e08.comprobar(reporte)
    if problemas:
        raise ErrorIntegracion("; ".join(problemas))
    if dir_candidata is not None:
        propios = {k: v for k, v in hashes_paquete(dir_candidata).items() if k == ARCHIVO_CONFIG}
        if reporte["accion"]["hashes"] != propios:
            raise ErrorIntegracion("el hash del reporte no corresponde al cgra.json de la candidata")
    metricas = dict(reporte["metricas"])
    metricas["estado"] = reporte["estado_evaluacion"]
    return {
        "A": reporte["accion"],
        "M": metricas,
        "C": reporte["costo_total"],
        "r": reporte["recompensa"],
        "e": reporte["estado_evaluacion"],
    }
