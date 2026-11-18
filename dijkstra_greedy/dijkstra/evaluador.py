"""Evaluador de configuraciones de la CGRA (solución Dijkstra + Greedy).

Es la caja "Evaluador" del ciclo de retroalimentación. Frente a la solución
RL solo cambia quién genera los ``PE*.txt``: el bloque Dijkstra en lugar de
los Skills 04–07. Todo lo demás sigue el contrato de ``solucion_rl.md``
(sección 10), con las piezas de ``contrato.py``:

    cgra.json
      validación previa (rl/agente/paquete.py)          -> accion_invalida
      por operación:
        Dijkstra mapea y escribe los PE*.txt            -> fallo_evaluador si no puede
        E03: mismas reglas que rl/medicion/validacion_programas.py -> fallo_evaluador
        simulador run_cgra.py en otro proceso, t_max    -> fallo_simulador
        validación funcional elemento por elemento      -> incorrecta | correcta
      C = promedio_op(K_A·C_A/C_A_ref + K_L·C_L,op/C_L_ref,op + K_R·C_R,op)
      (P_fallo si alguna operación no es correcta)

Los normalizadores son los congelados de ``normalizacion.json`` (medidos con
la solución RL). Una operación que no esté ahí se normaliza con la referencia
``M E M / M V M / M E M`` medida por este evaluador; si la referencia no es
correcta, no hay normalización (``ErrorReferencia``), como en el RL.

La recompensa ``r = C_best − C`` depende de la historia: la calcula el agente
al registrar la evaluación (``agente_greedy.py``).
"""

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import CARPETA_PROGRAMAS, RUTA_CGRA_JSON, cargar_grafo, evaluar_configuracion, leer_cgra
from . import operaciones as operaciones_del_c
from formato_cgra import FormatoCGRA  # noqa: E402 (cgra/src/ del simulador: lo agrega dijkstra/__init__.py)
from memoria import ImagenMemoria  # noqa: E402
from .contrato import (ACCION_INVALIDA, CORRECTA, ERROR_INFRAESTRUCTURA, FALLO_EVALUADOR, FALLO_SIMULADOR,
                       INCORRECTA, PENALIZADOS, PRIORIDAD_FALLA, RUTA_COSTOS, Costos, ErrorReferencia,
                       cargar_normalizadores, comparar, costo_reportado, fraccion_nops, validar_accion)
from .dfg import evaluar as evaluar_grafo
from .hardware import Compatibilidad, MallaMapeo
from .salida import escribir_pe
from .simulacion import CARPETA_PE, ejecutar as simular
from .validacion_programas import validar_programas

CONGELADOS = "congelados"


def _promedio(valores):
    valores = [v for v in valores if v is not None]
    return sum(valores) / len(valores) if valores else None


def configuracion_de(formato):
    """El cgra.json de tamaño fijo y malla explícita de un ``FormatoCGRA``."""
    return dict(filas=formato.filas, columnas=formato.columnas, palabras_memoria=formato.palabras_memoria,
                ancho_vector=formato.ancho_vector, tipos_pe=formato.tipos_pe,
                malla=[" ".join(f) for f in formato.malla])


@dataclass
class EvaluacionOperacion:
    """Resultado de una operación; ``C_L_norm`` y ``costo`` se llenan al normalizar."""

    operacion: str
    estado: str
    motivo: str = ""
    modo: str = None
    ciclos: int = None
    nops: int = None
    slots: int = None
    fraccion_nops: float = None
    C_L: float = None
    C_R: float = None
    validacion: dict = None
    errores: list = field(default_factory=list)
    C_L_norm: float = None
    costo: float = None
    nops_por_pe: dict = None  # PE -> NOPs, de la tabla "Instrucciones por PE" de reporte_ciclos.txt
    detalle: dict = None  # todo lo demás del reporte de ciclos (simulacion.leer_detalle)
    mapeo: dict = None  # métricas del mapeo de Dijkstra (operaciones, ciclos, instrucciones, transferencias, nops)
    tiempo_mapeo_s: float = None  # Dijkstra (placement y enrutamiento)
    tiempo_validacion_s: float = None  # PE*.txt y validación E03
    tiempo_simulacion_s: float = None  # simulador

    @property
    def valido(self):
        return self.estado == CORRECTA


@dataclass
class Evaluacion:
    """Estado que el evaluador le devuelve al agente (campos del reporte E08 del RL)."""

    mapa: list
    estado: str
    costo: float  # costo_total: el medido si es correcta, P_fallo si se penaliza, None si es infraestructura
    metricas: dict = field(default_factory=dict)  # C_A, C_L, C_R, C_A_norm, C_L_norm, ciclos, nops, ...
    motivo: str = ""
    operaciones: dict = field(default_factory=dict)  # operación -> EvaluacionOperacion
    errores: list = field(default_factory=list)
    normalizadores: dict = None
    costo_medido: float = None  # promedio de los costos por operación, aunque no sea correcta

    @property
    def valida(self):
        return self.estado == CORRECTA

    @property
    def area(self):
        return self.metricas.get("C_A")

    @property
    def latencia(self):
        return self.metricas.get("C_L_norm")

    @property
    def correctitud(self):
        return self.metricas.get("C_R")

    def como_dict(self):
        return asdict(self)


# -- de la corrida del simulador a la Evaluacion -------------------------------------
def medir_reporte(operacion, datos, grafo, regiones, costos, ruta_memoria=None):
    """``EvaluacionOperacion`` desde los datos de una corrida del simulador (``simulacion.leer_corrida``).

    El esperado es el grafo ejecutado en el orden del C sobre la memoria inicial;
    coincide con ``rl/medicion/entradas.py`` en las regiones de salida."""
    formato = Evaluador.formato(datos["configuracion"])
    ciclos, slots, nops = datos["ciclos"]["total"], datos["slots"], datos["nops"]
    n_pes = formato.filas * formato.columnas
    if ciclos < 0 or slots != ciclos * n_pes or not 0 <= nops <= slots:
        raise ValueError(f"{operacion}: métricas de reporte inconsistentes")
    if ruta_memoria is not None:
        inicial, _ = ImagenMemoria.leer(ruta_memoria).regiones()
        if inicial != datos["inicial"]:
            error = {"skill": "E05", "regla": "esperado_desligado",
                     "detalle": f"la memoria inicial del reporte no es {ruta_memoria}"}
            return EvaluacionOperacion(operacion, ERROR_INFRAESTRUCTURA, error["detalle"], errores=[error])
    todo = evaluar_grafo(grafo, datos["inicial"])
    esperado = {r: todo[r] for r in regiones}
    obtenido = {r: datos["final"][r] for r in regiones if r in datos["final"]}
    v = comparar(esperado, obtenido, costos.tolerancias)
    errores = [{"skill": "E05", "regla": "estructura", "detalle": t} for t in v["errores_estructurales"]]
    if v["elementos_fallidos"]:
        errores.append({"skill": "E05", "regla": "fuera_de_tolerancia",
                        "detalle": f"{v['elementos_fallidos']} de {v['total_elementos']} elementos fallan",
                        "primeros_fallos": v["primeros_fallos"]})
    estado = CORRECTA if v["aprobada"] else INCORRECTA
    validacion = {k: v[k] for k in ("aprobada", "elementos_fallidos", "total_elementos", "primeros_fallos")}
    return EvaluacionOperacion(
        operacion, estado, "; ".join(e["detalle"] for e in errores), datos.get("modo"), ciclos, nops, slots,
        fraccion_nops(ciclos, nops, n_pes), costos.latencia(ciclos, nops, n_pes), v["C_R"], validacion, errores,
        nops_por_pe=datos.get("nops_por_pe"), detalle=datos.get("detalle"))


def agregar(formato, operaciones, costos, normalizadores, motivo=""):
    """``Evaluacion`` de una configuración a partir de sus operaciones (rl/medicion/conjunto.py)."""
    estados_op = [e.estado for e in operaciones.values()]
    estado = next((c for c in PRIORIDAD_FALLA if c in estados_op), CORRECTA)
    c_a = costos.area(formato)
    k = costos.K
    for op, e in operaciones.items():
        if normalizadores is None or e.C_L is None:
            continue
        n = normalizadores["operaciones"][op]
        e.C_L_norm = e.C_L / n["C_L_ref"]
        if e.C_R is not None:
            e.costo = k["K_A"] * c_a / n["C_A_ref"] + k["K_L"] * e.C_L_norm + k["K_R"] * e.C_R
    ciclos = [e.ciclos for e in operaciones.values()]
    completas = None not in ciclos
    metricas = {
        "C_A": c_a,
        "C_A_norm": c_a / normalizadores["C_A_ref"] if normalizadores else None,
        # Como en el RL: con varias operaciones, C_L es la suma de los ciclos crudos; el valor
        # de la fórmula de cada operación está en operaciones[op].C_L.
        "C_L": sum(ciclos) if completas else None,
        "C_L_norm": _promedio([e.C_L_norm for e in operaciones.values()]) if completas else None,
        "C_R": _promedio([e.C_R for e in operaciones.values()]) if completas else None,
        "ciclos": sum(ciclos) if completas else None,
        "nops": sum(e.nops for e in operaciones.values()) if completas else None,
        "fraccion_nops": _promedio([e.fraccion_nops for e in operaciones.values()]),
        "capacidades": costos.conteo_capacidades(formato),
    }
    costos_op = [e.costo for e in operaciones.values()]
    medido = sum(costos_op) / len(costos_op) if costos_op and None not in costos_op else None
    errores = [{**x, "operacion": op} for op, e in operaciones.items() for x in e.errores]
    motivos = [motivo] if motivo else []
    motivos += [f"{op}: {e.motivo}" for op, e in operaciones.items() if e.motivo]
    return Evaluacion(formato.mapa(), estado, costo_reportado(estado, medido, costos.p_fallo), metricas,
                      "; ".join(motivos), operaciones, errores, normalizadores, medido)


def invalida(motivos, operaciones, costos, mapa=None):
    """``Evaluacion`` de una acción que no pasó la validación previa: no se mapea ni se simula."""
    detalle = "; ".join(motivos)
    ops = {op: EvaluacionOperacion(op, ACCION_INVALIDA, detalle) for op in operaciones}
    return Evaluacion(mapa or [], ACCION_INVALIDA, costos.p_fallo, {}, detalle, ops,
                      [{"skill": "A03", "regla": "accion_invalida", "detalle": m} for m in motivos])


class Evaluador:
    """Evalúa configuraciones de la CGRA con el bloque Dijkstra y el simulador."""

    def __init__(self, operaciones=None, carpeta_programas=CARPETA_PROGRAMAS, costos=None, opciones=None,
                 modo="auto", normalizadores=CONGELADOS, esquema=RUTA_CGRA_JSON):
        """``operaciones``: nombres de ``programas/`` (por defecto todas las de operaciones.c).
        Necesitan ``operaciones.ll`` y ``memoria.bin`` (skill 02). ``normalizadores``: los
        congelados de costos.json, un dict con el mismo formato, o ``None`` para medirlos
        con la referencia."""
        if operaciones is None:
            operaciones = operaciones_del_c()
        self.carpeta_programas = Path(carpeta_programas)
        self.costos = costos or Costos.cargar()
        self.compatibilidad = Compatibilidad.cargar()
        self.opciones = opciones
        self.modo = modo
        self.tipos_esquema = leer_cgra(esquema)["tipos_pe"]
        self.grafos = {op: cargar_grafo(op, self.carpeta_programas / op) for op in operaciones}
        # Regiones que escribe cada operación: ahí se mide el error (las mismas del RL).
        self.salidas = {op: sorted({n.banco for n in g.operaciones if n.op == "store"}) for op, g in self.grafos.items()}
        fijos = cargar_normalizadores(self.costos) if normalizadores == CONGELADOS else normalizadores
        self._fijos = fijos
        self._normalizadores = None
        self.referencia_medida = None  # la referencia con este evaluador (informativo)
        self.guardadas = {}  # clave -> Evaluacion
        self.mediciones = {}  # (clave, operación) -> EvaluacionOperacion

    # -- configuración ----------------------------------------------------------
    @staticmethod
    def formato(configuracion):
        """``FormatoCGRA`` desde un ``FormatoCGRA``, la ruta de un .json o un
        diccionario con las claves de cgra.json. Se valida igual que en el simulador."""
        if isinstance(configuracion, FormatoCGRA):
            return configuracion
        if isinstance(configuracion, dict):
            with tempfile.TemporaryDirectory() as directorio:
                ruta = Path(directorio) / "cgra.json"
                ruta.write_text(json.dumps(configuracion), encoding="utf-8")
                try:
                    return FormatoCGRA.cargar(ruta)
                except ValueError as error:
                    raise ValueError(str(error).replace(f"{ruta}: ", "")) from None
        return FormatoCGRA.cargar(configuracion)

    @staticmethod
    def datos(configuracion):
        if isinstance(configuracion, FormatoCGRA):
            return configuracion_de(configuracion)
        if isinstance(configuracion, dict):
            return configuracion
        return json.loads(Path(configuracion).read_text(encoding="utf-8"))

    def referencia(self):
        """El cgra.json de la referencia (costos.json → referencia.malla) con el esquema real."""
        malla = self.costos.referencia
        return dict(filas=len(malla), columnas=len(malla[0].split()),
                    palabras_memoria=self.costos.restricciones.get("palabras_memoria_max", 4096),
                    ancho_vector=(self.costos.restricciones.get("anchos_vector_permitidos") or [None])[0],
                    tipos_pe=self.tipos_esquema, malla=list(malla))

    def motivos_invalida(self, configuracion):
        """Validación previa; lista vacía si se puede evaluar."""
        try:
            formato = self.formato(configuracion)
        except (ValueError, TypeError, KeyError) as error:
            return [f"cgra.json: {error}"]
        return validar_accion(self.datos(configuracion), formato, self.costos.restricciones, self.tipos_esquema)

    def area(self, formato):
        """C_A de un ``FormatoCGRA``."""
        return self.costos.area(formato)

    @staticmethod
    def _clave(formato):
        tipos = {letra: sorted(t["instrucciones"]) for letra, t in formato.tipos_pe.items()}
        return json.dumps([formato.filas, formato.columnas, formato.palabras_memoria, formato.ancho_vector,
                           formato.malla, tipos], sort_keys=True)

    # -- normalizadores ---------------------------------------------------------
    def normalizadores(self):
        """Los congelados, completados con la referencia medida aquí para las operaciones que faltan."""
        if self._normalizadores is not None:
            return self._normalizadores
        fijos = self._fijos or {"operaciones": {}}
        faltan = [op for op in self.grafos if op not in fijos["operaciones"]]
        formato_ref = self.formato(self.referencia())
        c_a_ref = fijos.get("C_A_ref") or self.costos.area(formato_ref)
        por_op = {op: dict(fijos["operaciones"][op]) for op in self.grafos if op in fijos["operaciones"]}
        if faltan:
            medidas = {op: self._medir_operacion(op, formato_ref) for op in faltan}
            fallas = [f"{op}: {e.estado} {e.motivo}" for op, e in medidas.items() if e.estado != CORRECTA]
            if fallas:
                raise ErrorReferencia("la referencia no es correcta: " + "; ".join(fallas))
            for op, e in medidas.items():
                por_op[op] = {"C_A_ref": c_a_ref, "C_L_ref": e.C_L, "medido_con": "Dijkstra"}
        self._normalizadores = {"C_A_ref": c_a_ref, "operaciones": por_op, "referencia": "referencia",
                                "fuente": fijos.get("fuente")}
        return self._normalizadores

    # -- cotas (para que el agente no evalúe lo que no puede entrar a la cadena) ---
    def cota(self, configuracion):
        """Cota inferior del C de una configuración correcta: el término de área
        (latencia y C_R no son negativos). Las que no son correctas no entran a la cadena."""
        if min(self.costos.K.values()) < 0:
            return float("-inf")
        try:
            area = self.area(self.formato(configuracion))
        except (ValueError, TypeError, KeyError):
            return float("-inf")
        return self._termino_area(area)

    def cota_tamano(self, tipos_pe, pes):
        """Cota inferior de C para cualquier malla de ``pes`` PEs con estos tipos.
        Al menos un PE escribe en memoria (ST o VST); los demás pesan, como mínimo,
        lo del tipo más barato. Crece con ``pes``."""
        if min(self.costos.K.values()) < 0:
            return float("-inf")
        areas, escriben = [], []
        for tipo in tipos_pe.values():
            area = self.costos.area_pe(tipo["instrucciones"])
            areas.append(area)
            if {i.upper() for i in tipo["instrucciones"]} & {"ST", "VST"}:
                escriben.append(area)
        if not escriben:
            return float("inf")
        return self._termino_area(min(escriben) + (pes - 1) * min(areas))

    def _termino_area(self, area):
        fijos = self._fijos or {}
        c_a_ref = fijos.get("C_A_ref") or self.costos.area(self.formato(self.referencia()))
        return self.costos.K["K_A"] * area / c_a_ref

    # -- evaluación ---------------------------------------------------------------
    def evaluar(self, configuracion):
        """Costos de una configuración (``Evaluacion``)."""
        motivos = self.motivos_invalida(configuracion)
        if motivos:
            try:
                mapa = self.formato(configuracion).mapa()
            except (ValueError, TypeError, KeyError):
                mapa = None
            return invalida(motivos, self.grafos, self.costos, mapa)
        formato = self.formato(configuracion)
        clave = self._clave(formato)
        if clave not in self.guardadas:
            operaciones = {op: self._medir_operacion(op, formato) for op in self.grafos}
            simuladas = any(e.C_L is not None for e in operaciones.values())
            normalizadores = self.normalizadores() if simuladas else self._fijos
            # Cada evaluación tiene sus propias copias: agregar() les pone C_L_norm y costo.
            operaciones = {op: EvaluacionOperacion(**asdict(e)) for op, e in operaciones.items()}
            self.guardadas[clave] = agregar(formato, operaciones, self.costos, normalizadores)
        return self.guardadas[clave]

    def _medir_operacion(self, operacion, formato):
        clave = (self._clave(formato), operacion)
        if clave not in self.mediciones:
            self.mediciones[clave] = self._medir(operacion, formato)
        return self.mediciones[clave]

    def _medir(self, operacion, formato):
        """``_medir_etapas`` con el tiempo de cada etapa y las métricas del mapeo en la evaluación."""
        tiempos = {}
        e = self._medir_etapas(operacion, formato, tiempos)
        e.tiempo_mapeo_s = tiempos.get("mapeo")
        e.tiempo_validacion_s = tiempos.get("validacion")
        e.tiempo_simulacion_s = tiempos.get("simulacion")
        e.mapeo = tiempos.get("metricas_mapeo")
        return e

    def _medir_etapas(self, operacion, formato, tiempos):
        grafo = self.grafos[operacion]
        t0 = time.monotonic()
        resultado = evaluar_configuracion(grafo, formato, self.compatibilidad, self.opciones, self.modo)
        tiempos["mapeo"] = round(time.monotonic() - t0, 3)
        if resultado.valido:
            tiempos["metricas_mapeo"] = dict(resultado.metricas)
        if not resultado.valido:
            error = {"skill": "Dijkstra", "regla": "sin_mapeo", "detalle": resultado.motivo}
            return EvaluacionOperacion(operacion, FALLO_EVALUADOR, resultado.motivo, errores=[error])
        modo = resultado.metricas["modo"]
        ruta_memoria = self.carpeta_programas / operacion / "memoria.bin"
        if not ruta_memoria.is_file():
            error = {"skill": "E01", "regla": "infraestructura", "detalle": f"no existe {ruta_memoria}"}
            return EvaluacionOperacion(operacion, ERROR_INFRAESTRUCTURA, error["detalle"], modo, errores=[error])
        with tempfile.TemporaryDirectory() as directorio:
            t0 = time.monotonic()
            dir_pe = escribir_pe(resultado, MallaMapeo.desde_formato(formato), Path(directorio) / CARPETA_PE)
            e03 = validar_programas(formato, dir_pe, ruta_memoria, self.salidas[operacion])
            tiempos["validacion"] = round(time.monotonic() - t0, 3)
            if not e03["valida"]:
                estado = ERROR_INFRAESTRUCTURA if e03.get("fallo_infraestructura") else FALLO_EVALUADOR
                return EvaluacionOperacion(operacion, estado, "; ".join(e["detalle"] for e in e03["errores"][:5]),
                                           modo, errores=e03["errores"])
            t0 = time.monotonic()
            datos, error = simular(formato, dir_pe, ruta_memoria, directorio, self.costos.t_max_simulacion_s)
            tiempos["simulacion"] = round(time.monotonic() - t0, 3)
        if error is not None:
            return EvaluacionOperacion(operacion, FALLO_SIMULADOR, error["detalle"], modo, errores=[error])
        datos["modo"] = modo
        return medir_reporte(operacion, datos, grafo, self.salidas[operacion], self.costos, ruta_memoria)


__all__ = ["Costos", "Evaluacion", "EvaluacionOperacion", "Evaluador", "ErrorReferencia", "RUTA_COSTOS",
           "PENALIZADOS", "agregar", "configuracion_de", "invalida", "medir_reporte", "simular"]
