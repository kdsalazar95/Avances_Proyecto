"""Agente Greedy (solución Dijkstra + Greedy): camino por los PEs de una malla pesada.

El grafo de Greedy es la malla misma:

* **Vértices:** los PEs de la malla (``malla`` en agente.json; por defecto la más
  pesada, 32x32, con la distribución de cgra.json).
* **Aristas:** las conexiones de cada PE con sus vecinos norte, sur, este y oeste.
* **Peso de un PE:** ``w_tipo · (1 + NOPs / ciclos)``. ``w_tipo`` es el área de su
  tipo con los pesos w1-w4 de costos.json (M = 12, E = 1, V = 2); ``NOPs`` y
  ``ciclos`` salen del ``reporte_ciclos.txt`` del simulador, sumados sobre las
  operaciones. Un PE que pasa más tiempo en NOP pesa más.

Para tener los pesos, la malla se mapea con Dijkstra y se simula **una vez**
(en 32x32 tarda horas). Después, el **camino** (no recursivo, sin ciclos):
empieza en ``inicio`` (por defecto PE00, fila 0 y columna 0), revisa los vecinos
que todavía no visitó y elige el siguiente con una política **epsilon-greedy**:
con probabilidad ``epsilon`` decide el **agente RL** (un LLM que elige entre los
vecinos libres; ver ``rl_copilot.py``) y si no, **Dijkstra greedy** (el de menor peso).
Sin agente RL, la rama de RL es una exploración al azar. Repite hasta que el PE actual ya no tiene vecinos sin
visitar. Ningún PE se visita dos veces. Con ``epsilon = 0`` es Greedy puro; la
``semilla`` hace que el mismo camino se repita en cada corrida.

Resultado: los PEs del camino, con su tipo y su peso. No tienen que formar un
rectángulo; los PEs que no están en el camino quedan fuera. El camino se valúa
con la función de costo (``valuar_camino``) usando los datos de esa misma
simulación, sin volver a simular.
"""

import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import leer_cgra
from formato_cgra import generar_malla  # noqa: E402 (cgra/src/ del simulador)
from malla import Malla  # noqa: E402
from topologia import DESPLAZAMIENTO, DIRECCIONES  # noqa: E402

from .contrato import Costos

RUTA_PARAMETROS = Path(__file__).resolve().parents[1] / "config" / "agente.json"


@dataclass
class Parametros:
    malla: object = (32, 32)  # [filas, columnas] (distribución de cgra.json) o lista de filas "M E M"
    inicio: object = (0, 0)  # [fila, columna] del PE donde empieza el camino (PE00)
    epsilon: float = 0.1  # probabilidad de explorar (vecino libre al azar) en cada paso; 0 = Greedy puro
    semilla: int = 0  # semilla del azar de la exploración (mismo valor = mismo camino)
    procesos: int = None  # procesos para simular las operaciones en paralelo (None = una por operación)
    t_max_simulacion_s: float = None  # tiempo máximo del simulador para la malla (None = el de costos.json)
    rl: object = None  # agente RL del paso con probabilidad epsilon: {"comando": [...], "timeout_s": s}; None = al azar

    @classmethod
    def cargar(cls, ruta=RUTA_PARAMETROS):
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**{clave: datos[clave] for clave in cls.__dataclass_fields__ if clave in datos})


@dataclass
class PesoPE:
    """Peso de un vértice (PE) y los datos del reporte de ciclos con que se calculó."""

    pe: str
    tipo: str
    w_tipo: float
    nops: int
    ciclos: int
    peso: float


class Agente:
    """Epsilon-greedy sobre el grafo de PEs de una malla (determinístico para una misma semilla)."""

    def __init__(self, medir, base=None, parametros=None, costos=None, eleccion_rl=None):
        """``medir(configuración) -> Evaluacion``: el evaluador (Dijkstra + simulador); la
        Evaluacion debe traer ``nops_por_pe`` en cada operación. ``base``: el dict de cgra.json.
        ``eleccion_rl(actual, libres, pesos, camino) -> PE``: el agente RL; ``None`` si no hay."""
        self.medir = medir
        self.base = base if base is not None else leer_cgra()
        self.parametros = parametros or Parametros()
        self.costos = costos or Costos.cargar()
        self.eleccion_rl = eleccion_rl
        self.evaluacion = None  # la evaluación de la malla (una sola simulación)
        self.pesos = {}  # PE -> PesoPE
        self.camino = []  # PEs del camino, en orden
        # por paso: "inicio", "explotacion" (Dijkstra greedy), "rl" (agente RL), "rl_invalida"
        # (el agente RL no devolvió un vecino libre: se usa Dijkstra greedy) o "exploracion" (sin RL)
        self.decisiones = []

    # -- la malla ---------------------------------------------------------------------
    def de_distribucion(self, filas, columnas):
        """La malla de la distribución de cgra.json en ese tamaño."""
        if "distribucion" not in self.base:
            raise ValueError("la malla sale de 'distribucion' de cgra.json, y no la tiene")
        return tuple(tuple(fila) for fila in generar_malla(filas, columnas, self.base["distribucion"]))

    def malla(self):
        """La malla del grafo: ``[filas, columnas]`` con la distribución de cgra.json, o filas explícitas."""
        m = self.parametros.malla
        if m and isinstance(m[0], str):
            return tuple(tuple(fila.split()) for fila in m)
        return self.de_distribucion(m[0], m[1])

    def configuracion(self, malla=None):
        """La configuración que recibe el evaluador: cgra.json con tamaño fijo y ``malla`` explícita."""
        malla = malla or self.malla()
        datos = {k: v for k, v in self.base.items() if k not in ("distribucion", "malla", "descripcion")}
        datos.update(filas=len(malla), columnas=len(malla[0]), malla=[" ".join(fila) for fila in malla])
        return datos

    # -- pesos de los vértices --------------------------------------------------------
    def calcular_pesos(self, evaluacion, malla=None):
        """``{PE: PesoPE}`` con ``peso = w_tipo · (1 + NOPs/ciclos)``, sumando las operaciones."""
        malla = malla or self.malla()
        operaciones = list(evaluacion.operaciones.values())
        faltan = [o.operacion for o in operaciones if not o.nops_por_pe or not o.ciclos]
        if faltan:
            raise ValueError(f"sin reporte de ciclos por PE en: {', '.join(faltan)}")
        ciclos = sum(o.ciclos for o in operaciones)
        ids = Malla(len(malla), len(malla[0]))
        pesos = {}
        for f, fila in enumerate(malla):
            for c, tipo in enumerate(fila):
                pe = ids.id_de(f, c)
                w = self.costos.area_pe(self.base["tipos_pe"][tipo]["instrucciones"])
                nops = sum(o.nops_por_pe[pe] for o in operaciones)
                pesos[pe] = PesoPE(pe, tipo, w, nops, ciclos, w * (1 + nops / ciclos))
        return pesos

    # -- camino Greedy ----------------------------------------------------------------
    @staticmethod
    def vecinos(pe, filas, columnas):
        """Aristas de un PE: sus vecinos norte, sur, este y oeste que están dentro de la malla."""
        ids = Malla(filas, columnas)
        fila, columna = Malla.posicion(pe)
        resultado = []
        for direccion in DIRECCIONES:
            df, dc = DESPLAZAMIENTO[direccion]
            if 0 <= fila + df < filas and 0 <= columna + dc < columnas:
                resultado.append(ids.id_de(fila + df, columna + dc))
        return resultado

    def caminar(self, pesos, filas, columnas):
        """Desde ``inicio``, epsilon-greedy entre los vecinos sin visitar, hasta que el PE actual no
        tiene vecinos sin visitar. En cada paso, con probabilidad ``epsilon`` decide el agente RL
        (o, sin agente RL, uno de los libres al azar), y si no, Dijkstra greedy (el de menor peso;
        en empate, el primero fila por fila). Deja en ``self.decisiones`` qué se hizo en cada paso."""
        epsilon = self.parametros.epsilon
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon = {epsilon} debe estar entre 0 y 1")
        azar = random.Random(self.parametros.semilla)
        f0, c0 = self.parametros.inicio
        if not (0 <= f0 < filas and 0 <= c0 < columnas):
            raise ValueError(f"inicio ({f0}, {c0}) fuera de la malla {filas}x{columnas}")
        ids = Malla(filas, columnas)
        orden = {pe: i for i, pe in enumerate(ids.ids())}
        actual = ids.id_de(f0, c0)
        camino, visitados = [actual], {actual}
        self.decisiones = ["inicio"]
        while True:
            libres = [v for v in self.vecinos(actual, filas, columnas) if v not in visitados]
            if not libres:
                return camino
            if azar.random() < epsilon:
                if self.eleccion_rl is None:
                    actual = azar.choice(sorted(libres, key=orden.get))
                    self.decisiones.append("exploracion")
                else:
                    propuesta = self.eleccion_rl(actual, sorted(libres, key=orden.get), self.pesos, camino)
                    if propuesta in libres:
                        actual = propuesta
                        self.decisiones.append("rl")
                    else:
                        actual = min(libres, key=lambda v: (self.pesos[v].peso, orden[v]))
                        self.decisiones.append("rl_invalida")
            else:
                actual = min(libres, key=lambda v: (pesos[v].peso, orden[v]))
                self.decisiones.append("explotacion")
            camino.append(actual)
            visitados.add(actual)

    def buscar(self, evaluacion=None):
        """Simula la malla (o usa ``evaluacion`` si ya se tiene), calcula los pesos y recorre el camino.
        Devuelve la lista de PEs del camino."""
        malla = self.malla()
        self.evaluacion = evaluacion if evaluacion is not None else self.medir(self.configuracion(malla))
        if not self.evaluacion.valida:
            raise ValueError(f"la malla {len(malla)}x{len(malla[0])} no es correcta "
                             f"({self.evaluacion.estado}): {self.evaluacion.motivo}")
        self.pesos = self.calcular_pesos(self.evaluacion, malla)
        self.camino = self.caminar(self.pesos, len(malla), len(malla[0]))
        return self.camino

    # -- valuación del camino -----------------------------------------------------------
    def valuar_camino(self):
        """Costo de los PEs del camino con la función de costo (contrato.py), sin volver a simular:
        los ciclos, los NOPs de cada PE y C_R salen de la evaluación de la malla.

        C_A = Σ w_tipo de los PEs del camino; por operación, C_L = latencia(ciclos, NOPs del
        camino, PEs del camino); C = promedio de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R."""
        n = self.evaluacion.normalizadores
        if not n:
            raise ValueError("la evaluación de la malla no trae normalizadores")
        k = self.costos.K
        pes = len(self.camino)
        c_a = sum(self.pesos[pe].w_tipo for pe in self.camino)
        por_operacion = {}
        for op, e in self.evaluacion.operaciones.items():
            nops = sum(e.nops_por_pe[pe] for pe in self.camino)
            c_l = self.costos.latencia(e.ciclos, nops, pes)
            ref = n["operaciones"][op]
            c_l_norm = c_l / ref["C_L_ref"]
            costo = k["K_A"] * c_a / ref["C_A_ref"] + k["K_L"] * c_l_norm + k["K_R"] * e.C_R
            por_operacion[op] = {"ciclos": e.ciclos, "nops": nops, "C_L": c_l, "C_L_norm": c_l_norm,
                                 "C_R": e.C_R, "costo": costo}
        costos_op = [o["costo"] for o in por_operacion.values()]
        return {
            "C": sum(costos_op) / len(costos_op),
            "C_A": c_a,
            "C_A_norm": c_a / n["C_A_ref"],
            "C_L_norm": sum(o["C_L_norm"] for o in por_operacion.values()) / len(por_operacion),
            "C_R": sum(o["C_R"] for o in por_operacion.values()) / len(por_operacion),
            "pes": pes,
            "por_operacion": por_operacion,
            "formula": "C = promedio de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R; C_A = Σ w_tipo del camino; "
                       "C_L = ciclos·(1 + NOPs del camino/(ciclos·PEs del camino)); ciclos, NOPs y C_R de la "
                       "evaluación de la malla (sin volver a simular)",
        }

    # -- resultado --------------------------------------------------------------------
    def resultado(self):
        """Los PEs del camino con su tipo y peso, y el mapa de la malla con '.' en los que quedan fuera."""
        malla = self.malla()
        ids = Malla(len(malla), len(malla[0]))
        en_camino = set(self.camino)
        mapa = [" ".join(t if ids.id_de(f, c) in en_camino else "." for c, t in enumerate(fila))
                for f, fila in enumerate(malla)]
        tipos = {}
        for pe in self.camino:
            tipos[self.pesos[pe].tipo] = tipos.get(self.pesos[pe].tipo, 0) + 1
        return {
            "inicio": self.camino[0] if self.camino else None,
            "pes": len(self.camino),
            "por_tipo": tipos,
            "peso_total": sum(self.pesos[pe].peso for pe in self.camino),
            "valuacion": self.valuar_camino(),
            "epsilon": self.parametros.epsilon,
            "semilla": self.parametros.semilla,
            "exploraciones": self.decisiones.count("exploracion"),
            "rl": self.decisiones.count("rl"),
            "rl_invalidas": self.decisiones.count("rl_invalida"),
            "camino": [{"paso": i, "decision": d, **asdict(self.pesos[pe])}
                       for i, (pe, d) in enumerate(zip(self.camino, self.decisiones))],
            "mapa": mapa,
        }


# -- simulación de la malla: una operación por proceso ---------------------------------
def _costos(ruta_costos, t_max):
    costos = Costos.cargar(ruta_costos)
    if t_max is not None:
        costos.t_max_simulacion_s = t_max
    return costos


def _evaluar_operacion(argumentos):
    from .evaluador import Evaluador
    operacion, carpeta, ruta_costos, t_max, configuracion = argumentos
    evaluador = Evaluador([operacion], carpeta, _costos(ruta_costos, t_max))
    return evaluador.evaluar(configuracion).operaciones[operacion]


def medir_en_paralelo(configuracion, operaciones, carpeta, ruta_costos, t_max=None, procesos=None):
    """Evalúa la configuración con cada operación en su propio proceso y junta los costos
    como ``Evaluador.evaluar`` (mismos normalizadores y ecuaciones)."""
    from .evaluador import EvaluacionOperacion, Evaluador, agregar
    costos = _costos(ruta_costos, t_max)
    evaluador = Evaluador(list(operaciones), carpeta, costos)
    if evaluador.motivos_invalida(configuracion):
        return evaluador.evaluar(configuracion)
    argumentos = [(op, str(carpeta), str(ruta_costos), t_max, configuracion) for op in operaciones]
    with ProcessPoolExecutor(procesos or len(operaciones)) as pool:
        por_op = dict(zip(operaciones, pool.map(_evaluar_operacion, argumentos)))
    por_op = {op: EvaluacionOperacion(**asdict(e)) for op, e in por_op.items()}
    simuladas = any(e.C_L is not None for e in por_op.values())
    normalizadores = evaluador.normalizadores() if simuladas else evaluador._fijos
    return agregar(Evaluador.formato(configuracion), por_op, costos, normalizadores)
