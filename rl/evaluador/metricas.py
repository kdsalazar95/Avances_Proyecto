"""E06 · Métricas: C_A (área proxy), C_L (latencia), C_R y costo total.

C_A = w1·n_MS + w2·n_MV + w3·n_CS + w4·n_CV, contando las capacidades que
declara cada tipo de PE en su lista de instrucciones (proxy aditivo):

    MS: LD o ST          MV: VLD o VST
    CS: ADD/SUB/MUL/DIV  CV: VADD/VSUB/VMUL/VDIV

Así M (LD/ST/VLD/VST) aporta MS+MV, E aporta CS y V aporta CV.

C_L según ``formula_latencia``:

- ``"ciclos"``: ciclos totales.
- ``"ciclos_x_nops"``: ciclos·(1 + fracción de NOP), con fracción = NOPs/(ciclos·PEs). Es el criterio
  del profesor ("¿cuántos ciclos le toma hacer cómputo? ¿cuántos NOPs/total hay? Entre más NOPs, mayor
  es el costo"): sin NOPs vale los ciclos; con la mitad de las ranuras en NOP, 1.5·ciclos. Equivale a
  ciclos + NOPs promedio por PE.

C = K_A·C_A/C_A,ref + K_L·C_L/C_L,ref + K_R·C_R
"""

from ..comun import usar_simulador

usar_simulador()
from formato_cgra import FormatoCGRA  # noqa: E402

CAPACIDADES = {
    "MS": {"LD", "ST"},
    "MV": {"VLD", "VST"},
    "CS": {"ADD", "SUB", "MUL", "DIV"},
    "CV": {"VADD", "VSUB", "VMUL", "VDIV"},
}
PESO_DE = {"MS": "w1_MS", "MV": "w2_MV", "CS": "w3_CS", "CV": "w4_CV"}


class ErrorMetricas(ValueError):
    """Falta un parámetro o un normalizador: no se usan valores ocultos."""


def capacidades_tipo(instrucciones):
    instrucciones = {op.upper() for op in instrucciones}
    return [cap for cap, ops in CAPACIDADES.items() if instrucciones & ops]


def conteo_capacidades(formato):
    """``{"MS": n, "MV": n, "CS": n, "CV": n}`` de la malla de un FormatoCGRA."""
    conteo = dict.fromkeys(CAPACIDADES, 0)
    for fila in formato.malla:
        for letra in fila:
            for cap in capacidades_tipo(formato.tipos_pe[letra]["instrucciones"]):
                conteo[cap] += 1
    return conteo


def costo_area(conteo, pesos):
    faltan = [PESO_DE[c] for c in CAPACIDADES if PESO_DE[c] not in pesos]
    if faltan:
        raise ErrorMetricas(f"faltan pesos de área: {', '.join(faltan)}")
    return sum(pesos[PESO_DE[cap]] * n for cap, n in conteo.items())


def numero_pes(ruta_cgra):
    formato = FormatoCGRA.cargar(ruta_cgra)
    return formato.filas * formato.columnas


def costo_area_json(ruta_cgra, pesos):
    """C_A directamente desde un cgra.json (no necesita simular)."""
    conteo = conteo_capacidades(FormatoCGRA.cargar(ruta_cgra))
    return costo_area(conteo, pesos), conteo


FORMULAS_LATENCIA = ("ciclos", "ciclos_x_nops")


def fraccion_nops(ciclos, nops, n_pes):
    """NOPs / (ciclos·PEs): proporción de ranuras de instrucción en NOP."""
    if ciclos is None or nops is None or not n_pes or ciclos <= 0:
        return None
    return nops / (ciclos * n_pes)


def costo_latencia(ciclos, nops, formula, n_pes=None):
    if formula not in FORMULAS_LATENCIA:
        raise ErrorMetricas(f"fórmula de latencia no soportada: {formula}")
    if ciclos is None:
        return None
    if formula == "ciclos":
        return ciclos
    fraccion = fraccion_nops(ciclos, nops, n_pes)
    if fraccion is None:
        raise ErrorMetricas("'ciclos_x_nops' necesita los NOPs y el número de PEs")
    return ciclos * (1 + fraccion)


def costo_total(c_a, c_l, c_r, K, normalizadores):
    """Costo normalizado. Devuelve ``(costo, C_A_norm, C_L_norm)``; ``None`` si falta una métrica."""
    for clave in ("K_A", "K_L", "K_R"):
        if clave not in K:
            raise ErrorMetricas(f"falta {clave}")
    if not normalizadores or normalizadores.get("C_A_ref", 0) <= 0 or normalizadores.get("C_L_ref", 0) <= 0:
        raise ErrorMetricas("faltan normalizadores válidos de la referencia (C_A_ref > 0, C_L_ref > 0)")
    c_a_norm = None if c_a is None else c_a / normalizadores["C_A_ref"]
    c_l_norm = None if c_l is None else c_l / normalizadores["C_L_ref"]
    if c_a_norm is None or c_l_norm is None or c_r is None:
        return None, c_a_norm, c_l_norm
    return K["K_A"] * c_a_norm + K["K_L"] * c_l_norm + K["K_R"] * c_r, c_a_norm, c_l_norm


def calcular(ruta_cgra, ciclos, nops, c_r, parametros, normalizadores):
    """Todas las métricas de E06 para una acción simulada."""
    c_a, conteo = costo_area_json(ruta_cgra, parametros["pesos_area"])
    n_pes = numero_pes(ruta_cgra)
    c_l = costo_latencia(ciclos, nops, parametros["formula_latencia"], n_pes)
    costo, c_a_norm, c_l_norm = costo_total(c_a, c_l, c_r, parametros["K"], normalizadores)
    return {
        "C_A": c_a, "C_L": c_l, "C_R": c_r,
        "C_A_norm": c_a_norm, "C_L_norm": c_l_norm,
        "ciclos": ciclos, "nops": nops,
        "fraccion_nops": fraccion_nops(ciclos, nops, n_pes),
        "capacidades": conteo,
        "costo_total": costo,
        "formula_latencia": parametros["formula_latencia"],
    }
