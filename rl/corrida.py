"""Parámetros experimentales de una corrida (corrida.json).

Los límites de búsqueda, pesos, K, tolerancias, penalización y referencia
viven aquí, no en el ``cgra.json`` de hardware. Si falta un parámetro
obligatorio se devuelve un error claro: no hay valores ocultos en el código.
"""

from pathlib import Path

from .comun import ESQUEMA_CGRA, FUENTE_OPERACIONES, RAIZ_REPO, leer_json

OPERACIONES = ("reduccion", "matmul", "convolucion")
FORMULAS_LATENCIA = ("ciclos", "ciclos_x_nops")

OBLIGATORIOS = {
    "id": str,
    "pesos_area": dict,
    "K": dict,
    "formula_latencia": str,
    "tolerancias": dict,
    "penalizacion": dict,
    "tolerancia_empate_costo": (int, float),
    "limites": dict,
    "referencia": dict,
    "compilador": dict,
}
PESOS = ("w1_MS", "w2_MV", "w3_CS", "w4_CV")
KS = ("K_A", "K_L", "K_R")
TOLERANCIAS = ("eps_abs", "eps_rel", "epsilon_cr")
LIMITES = ("max_iteraciones", "paciencia", "t_max_simulacion_s")


class ErrorCorrida(ValueError):
    """Parámetros de corrida incompletos o incoherentes."""


def _numero(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_corrida(datos):
    """Devuelve la lista de errores de un corrida.json (vacía si es válido)."""
    errores = []
    for campo, tipo in OBLIGATORIOS.items():
        if campo not in datos:
            errores.append(f"falta '{campo}'")
        elif not isinstance(datos[campo], tipo) or isinstance(datos[campo], bool):
            errores.append(f"'{campo}' tiene un tipo inválido")
    if errores:
        return errores
    if "operaciones" in datos:
        ops = datos["operaciones"]
        if not isinstance(ops, list) or not ops or len(set(ops)) != len(ops) or \
                any(op not in OPERACIONES for op in ops):
            errores.append(f"'operaciones' debe ser una lista sin repetir de {OPERACIONES}")
    elif datos.get("operacion") not in OPERACIONES:
        errores.append(f"falta 'operaciones' (lista) u 'operacion' (una de {OPERACIONES})")
    for grupo, claves in (("pesos_area", PESOS), ("K", KS), ("tolerancias", TOLERANCIAS)):
        for clave in claves:
            valor = datos[grupo].get(clave)
            if not _numero(valor) or valor < 0:
                errores.append(f"'{grupo}.{clave}' debe ser un número >= 0 (valor: {valor!r})")
    if datos["formula_latencia"] not in FORMULAS_LATENCIA:
        errores.append(f"formula_latencia '{datos['formula_latencia']}' no soportada: {FORMULAS_LATENCIA}")
    p_fallo = datos["penalizacion"].get("P_fallo")
    if not _numero(p_fallo) or p_fallo <= 0:
        errores.append("'penalizacion.P_fallo' debe ser un número > 0")
    if datos["tolerancia_empate_costo"] < 0:
        errores.append("'tolerancia_empate_costo' debe ser >= 0")
    for clave in LIMITES:
        valor = datos["limites"].get(clave)
        if not _numero(valor) or valor <= 0:
            errores.append(f"'limites.{clave}' debe ser un número > 0")
    referencia = datos["referencia"]
    if "cgra" not in referencia and not isinstance(referencia.get("generar", {}).get("malla"), list):
        errores.append("'referencia' necesita 'cgra' (ruta a un cgra.json) o 'generar.malla' (lista de filas)")
    compilador = datos["compilador"]
    if compilador.get("tipo") not in ("comando", "reduccion_mcm"):
        errores.append("'compilador.tipo' debe ser 'comando' (Skills 01–09 con un LLM) o 'reduccion_mcm' (sustituto)")
    elif compilador["tipo"] == "comando" and not compilador.get("comando"):
        errores.append("'compilador.comando' es obligatorio con tipo 'comando'")
    n_corr = compilador.get("N_corr", 2)
    if not isinstance(n_corr, int) or isinstance(n_corr, bool) or n_corr < 0:
        errores.append("'compilador.N_corr' debe ser un entero >= 0")
    return errores


def cargar_corrida(ruta):
    """Lee y valida corrida.json; completa rutas por defecto (fuente C y esquema)."""
    ruta = Path(ruta)
    try:
        datos = leer_json(ruta)
    except FileNotFoundError:
        raise ErrorCorrida(f"No existe {ruta}") from None
    errores = validar_corrida(datos)
    if errores:
        raise ErrorCorrida(f"{ruta}: " + "; ".join(errores))
    datos.setdefault("fuente_c", str(FUENTE_OPERACIONES.relative_to(RAIZ_REPO)))
    datos.setdefault("esquema_cgra", str(ESQUEMA_CGRA.relative_to(RAIZ_REPO)))
    datos.setdefault("restricciones", {})
    datos.setdefault("politica", {"tipo": "manual"})
    datos["limites"].setdefault("t_max_total_s", None)
    return datos


def es_conjunto(corrida):
    """True si la corrida evalúa cada configuración con varias operaciones ('operaciones')."""
    return "operaciones" in corrida


def resolver(ruta):
    """Rutas relativas de corrida.json se interpretan desde la raíz del repositorio."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta
