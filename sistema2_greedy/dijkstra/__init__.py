"""Bloque Dijkstra del evaluador de la CGRA: sustituye los skills 04 a 07.

Mismo grafo y misma malla -> siempre el mismo mapeo. Uso típico (el que
necesita el agente Greedy: el grafo se calcula una vez por operación y se
reutiliza en todas las configuraciones)::

    from dijkstra import cargar_grafo, evaluar_configuracion
    grafo = cargar_grafo("matmul")                                       # skills 01-04, una vez
    formato = FormatoCGRA.cargar(filas=3, columnas=3)                     # cgra.json da rangos
    resultado = evaluar_configuracion(grafo, formato)                     # skills 05-07 + verificador
    resultado.valido, resultado.motivo, resultado.metricas

Partes:

* ``ir_llvm``      skill 04: ``operaciones.ll`` -> grafo de datos desenrollado
* ``vectorizar``   agrupa de a ``ancho_vector`` operaciones (VLD, VADD, VBCAST, VREDSUM...)
* ``hardware``     skills 05-06: tabla de compatibilidad y malla desde cgra.json
* ``mapeador``     skill 07: Dijkstra en la malla extendida en el tiempo
* ``verificador``  revisión independiente del mapeo
* ``salida``       PE*.txt (formato del skill 08) y corrida en el simulador
* ``evaluador``    el Evaluador del ciclo agente Greedy <-> evaluador: C_A, C_L, C_R y C
* ``agente``       el agente Greedy (camino por los PEs de menor costo C) que usa el evaluador
"""

import sys
from pathlib import Path

# El simulador es el de cgra/src/ (compartido con la solución RL, sin modificar).
# Dijkstra + Greedy vive aparte, en sistema2_greedy/.
RAIZ_DIJKSTRA_GREEDY = Path(__file__).resolve().parents[1]
RAIZ_PROYECTO = RAIZ_DIJKSTRA_GREEDY.parent
CARPETA_CONFIG = RAIZ_DIJKSTRA_GREEDY / "config"
RAIZ_CGRA = RAIZ_PROYECTO / "cgra"
SRC_SIMULADOR = RAIZ_CGRA / "src"
if str(SRC_SIMULADOR) not in sys.path:
    sys.path.insert(0, str(SRC_SIMULADOR))

from formato_cgra import FormatoCGRA  # noqa: E402
from memoria import ImagenMemoria  # noqa: E402

from .dfg import GrafoDatos, Operacion, evaluar
from .hardware import Compatibilidad, MallaMapeo
from .ir_llvm import ErrorIR, grafo_desde_ir
from .mapeador import Opciones, ResultadoMapeo, mapear
from .vectorizar import vectorizar
from .verificador import verificar

MODOS = ("auto", "escalar", "vectorial", "mejor")
ARITMETICA_VECTORIAL = frozenset({"VADD", "VSUB", "VMUL", "VDIV"})

CARPETA_PROGRAMAS = RAIZ_CGRA / "programas"
FUENTE_OPERACIONES = RAIZ_CGRA / "operaciones" / "operaciones.c"
RUN_CGRA = SRC_SIMULADOR / "run_cgra.py"
RUTA_CGRA_JSON = SRC_SIMULADOR / "cgra.json"


def operaciones(fuente=FUENTE_OPERACIONES):
    """Las operaciones de la tabla OPERACIONES de operaciones.c, en su orden."""
    from run_cgra import SimuladorCGRA
    return SimuladorCGRA.operaciones(fuente)


def leer_cgra(ruta=RUTA_CGRA_JSON):
    """El cgra.json como diccionario, sin validar."""
    import json
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def rango(valor):
    """``filas``/``columnas`` de cgra.json como ``(minimo, maximo)``: un número fijo ``n``
    es ``(n, n)``; un rango es ``{"minimo": .., "maximo": ..}`` (sin máximo, 32)."""
    if isinstance(valor, dict):
        return valor.get("minimo", 1), valor.get("maximo", 32)
    return valor, valor


def cargar_grafo(operacion, carpeta=None, ruta_memoria=None, reenvio=True):
    """Grafo de ``@<operacion>`` desde ``programas/<operacion>/operaciones.ll``
    y las regiones de su ``memoria.bin`` (salidas de los skills 02-03)."""
    carpeta = Path(carpeta) if carpeta else CARPETA_PROGRAMAS / operacion
    _, formas = ImagenMemoria.leer(ruta_memoria or carpeta / "memoria.bin").regiones()
    return grafo_desde_ir(carpeta / "operaciones.ll", operacion, formas, reenvio)


def grafo_vectorial(grafo, ancho):
    """El grafo vectorizado de a ``ancho``, calculado una vez y guardado en el grafo."""
    if ancho not in grafo.vectorizados:
        grafo.vectorizados[ancho] = vectorizar(grafo, ancho)
    return grafo.vectorizados[ancho]


def evaluar_configuracion(grafo, formato, compatibilidad=None, opciones=None, modo="auto"):
    """Mapea el grafo en la CGRA de ``formato`` (un ``FormatoCGRA`` o la ruta
    de un .json) y lo verifica. Nunca lanza por una configuración mala: el
    resultado es inválido y ``motivo`` dice por qué.

    ``modo``: ``"escalar"`` mapea el grafo tal cual; ``"vectorial"`` lo
    vectoriza con el ``ancho_vector`` de la CGRA; ``"auto"`` (por defecto)
    usa el vectorial si la CGRA tiene aritmética vectorial y el mapeo sale
    válido, y si no el escalar; ``"mejor"`` prueba los dos y se queda con el
    de menos ciclos. ``metricas["modo"]`` dice cuál se usó.
    """
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo} (opciones: {', '.join(MODOS)})")
    if not isinstance(formato, FormatoCGRA):
        try:
            formato = FormatoCGRA.cargar(formato)
        except ValueError as error:
            return ResultadoMapeo.invalido(f"formato de la CGRA inválido: {error}", len(grafo))
    compatibilidad = compatibilidad or Compatibilidad.cargar()
    malla = MallaMapeo.desde_formato(formato)
    tiene_vectores = bool(formato.ancho_vector) and any(
        ARITMETICA_VECTORIAL & pe.instrucciones for pe in malla.pes)

    intentos = []
    if modo in ("vectorial", "auto", "mejor") and tiene_vectores:
        intentos.append(("vectorial", grafo_vectorial(grafo, formato.ancho_vector)))
    elif modo == "vectorial":
        return ResultadoMapeo.invalido("la CGRA no tiene ancho_vector ni PEs con aritmética vectorial", len(grafo))
    if modo in ("escalar", "mejor") or modo == "auto" and not tiene_vectores:
        intentos.append(("escalar", grafo))

    mejor = None
    for nombre, actual in intentos:
        resultado = _mapear_y_verificar(actual, malla, compatibilidad, opciones)
        resultado.metricas["modo"] = nombre
        if modo == "auto" and nombre == "vectorial" and not resultado.valido:
            # Sin ruta para los vectores: se intenta escalar.
            resultado = _mapear_y_verificar(grafo, malla, compatibilidad, opciones)
            resultado.metricas["modo"] = "escalar"
        if mejor is None or resultado.valido and (
                not mejor.valido or resultado.metricas["ciclos"] < mejor.metricas["ciclos"]):
            mejor = resultado
    return mejor


def _mapear_y_verificar(grafo, malla, compatibilidad, opciones):
    resultado = mapear(grafo, malla, compatibilidad, opciones)
    if resultado.valido:
        errores = verificar(grafo, malla, compatibilidad, resultado)
        if errores:
            return ResultadoMapeo.invalido(f"el verificador rechazó el mapeo: {'; '.join(errores[:5])}", len(grafo))
    return resultado


__all__ = [
    "CARPETA_PROGRAMAS", "FUENTE_OPERACIONES", "RUN_CGRA", "RUTA_CGRA_JSON", "Compatibilidad", "ErrorIR", "GrafoDatos", "MallaMapeo", "Operacion", "Opciones",
    "MODOS", "ResultadoMapeo", "cargar_grafo", "evaluar", "evaluar_configuracion", "grafo_desde_ir", "grafo_vectorial", "leer_cgra", "operaciones", "rango",
    "mapear", "vectorizar", "verificar",
]
