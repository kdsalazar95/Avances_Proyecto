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
con probabilidad ``epsilon`` **explora** (un vecino libre al azar; este sistema no usa LLM)
y si no, **greedy**: el vecino que deja el menor
costo C del camino (la función de costo del PDF, ``valuar_camino``; un camino que no cumple
los tipos de PE cuesta ``P_fallo``, como en el contrato). **Paro:** el camino termina cuando ya cumple los tipos de PE y ningún vecino
libre baja su C ("si el costo sube, se descarta"), o cuando el PE actual ya no tiene vecinos sin
visitar. **Seguir o parar:** en el punto de paro, con probabilidad ``epsilon`` el azar decide
si el camino para o sigue ``n`` PEs más (``1 ≤ n ≤ max_extension``): el
estimado siempre prefiere menos PEs (su latencia es la de la malla completa), y así se pueden
probar configuraciones más grandes.
Durante esos ``n`` pasos no se revisa el paro; después se revisa otra vez.
Ningún PE se visita dos veces. Con ``epsilon = 0`` es Greedy puro; la
``semilla`` hace que el mismo camino se repita en cada corrida. En cada paso se guarda la
recompensa ``r_t = C(t−1) − C(t)`` (positiva si el paso bajó el costo).

**Episodio** (``evaluar_episodio``): el camino terminado se convierte en una malla simulable, el
rectángulo que lo contiene (``malla_episodio``: sus filas y columnas, sin las columnas de memoria,
con una columna M al oeste y otra al este), y esa malla se simula con el mismo evaluador. Su C es
el costo **real** del episodio, y la recompensa del episodio es ``R = C_best_prev − C`` (como en la
solución RL), con ``C_best_prev`` = el mejor C real de los episodios anteriores (al principio, el
C de la malla completa).

**Ciclo de episodios** (``correr``): se repiten episodios hasta ``max_episodios`` o hasta que pasen
``paciencia`` episodios seguidos sin que el mejor C baje. Cada episodio usa la semilla
``semilla + n`` (si no, todos darían el mismo camino). Si el C real de un episodio es menor que el
mejor, ese episodio pasa a ser el mejor; si no, se descarta (pág. 5 del PDF). Un rectángulo ya
simulado no se vuelve a simular (``repetido``). El C real de cada episodio es la retroalimentación
del evaluador: decide si el episodio pasa a ser el mejor y cuándo se acaba la paciencia.

Resultado: los PEs del camino, con su tipo y su peso. No tienen que formar un
rectángulo; los PEs que no están en el camino quedan fuera. El camino se valúa
con la función de costo (``valuar_camino``) usando los datos de esa misma
simulación, sin volver a simular.
"""

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import leer_cgra
from formato_cgra import generar_malla  # noqa: E402 (cgra/src/ del simulador)
from malla import Malla  # noqa: E402
from topologia import DESPLAZAMIENTO, DIRECCIONES  # noqa: E402

from .contrato import (COMPUTO_OPS, CORRECTA, ESCALAR_OPS, MEMORIA_OPS, NO_ATRIBUIBLES, VECTORIAL_OPS, Costos,
                       recompensa)
from .mapeador import Opciones

RUTA_PARAMETROS = Path(__file__).resolve().parents[1] / "config" / "agente.json"


@dataclass
class Parametros:
    malla: object = (32, 32)  # [filas, columnas] (distribución de cgra.json) o lista de filas "M E M"
    inicio: object = (0, 0)  # [fila, columna] del PE donde empieza el camino (PE00)
    epsilon: float = 0.3  # probabilidad de explorar (acción al azar) en cada paso; 0 = Greedy puro
    semilla: int = 0  # semilla del azar de la exploración (mismo valor = mismo camino)
    procesos: int = None  # procesos para simular las operaciones en paralelo (None = una por operación)
    mapeador_procesos: int = 1  # procesos que prueban los candidatos del mapeador (1 = secuencial; mismo resultado)
    t_max_simulacion_s: float = None  # tiempo máximo del simulador para la malla (None = el de costos.json)
    max_episodios: int = 10  # tope de episodios por corrida (como max_iteraciones de rl/)
    paciencia: int = 3  # episodios seguidos sin que baje el mejor C para terminar (como en rl/)
    max_extension: int = 10  # PEs que el azar puede pedir de más en el punto de paro
    modo: str = "camino"  # "camino": crecer desde PE00 · "bloques": reducir la malla de referencia por bloques
    bloques: object = (16, 8, 4, 2, 1)  # tamaños de bloque del modo "bloques", de mayor a menor
    tipos: bool = False  # modo "bloques": después del descenso, fase de tipos (cambiar PEs interiores E <-> V)
    max_cambios_tipo: int = 40  # tope de mallas simuladas en la fase de tipos de un episodio

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

    def __init__(self, medir, base=None, parametros=None, costos=None):
        """``medir(configuración) -> Evaluacion``: el evaluador (Dijkstra + simulador); la
        Evaluacion debe traer ``nops_por_pe`` en cada operación. ``base``: el dict de cgra.json."""
        self.medir = medir
        self.base = base if base is not None else leer_cgra()
        self.parametros = parametros or Parametros()
        self.costos = costos or Costos.cargar()
        self.evaluacion = None  # la evaluación de la malla (una sola simulación)
        self.pesos = {}  # PE -> PesoPE
        self.camino = []  # PEs del camino, en orden
        # por paso: "inicio", "explotacion" (greedy: menor C) o "exploracion" (al azar)
        self.decisiones = []
        # por qué terminó el camino: "sin_mejora" (cumple los tipos y ningún vecino baja C) o
        # "sin_vecinos" (el PE actual no tiene vecinos sin visitar)
        self.parada = None
        # decisiones en los puntos de paro: {"paso", "pes", "decision" (exploracion),
        # "n" (0 = parar; n > 0 = seguir n PEs más)}
        self.extensiones = []
        self.evaluacion_episodio = None  # la simulación del rectángulo del camino (evaluar_episodio)
        self.episodio = None  # resumen del episodio: malla, estado, C real, recompensa
        self.semilla_actual = self.parametros.semilla  # la semilla del episodio en curso
        self.n_episodio = 0  # número del episodio en curso (0, 1, ...)
        self.historial = []  # episodios ya evaluados, en orden (correr)
        self.mejor = None  # el episodio correcto de menor C real
        self.c_best = None  # C_best: el mejor C real hasta ahora (al principio, el de la malla completa)
        self.motivo_fin = None  # por qué terminó correr: "max_episodios" o "paciencia"
        self.cache_episodios = {}  # clave de la configuración -> Evaluacion (rectángulos ya simulados)
        self._vistos = set()  # claves de los rectángulos que ya salieron en esta corrida (correr)
        self.al_simular = None  # al_simular(configuracion, evaluacion): tras cada simulación nueva

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

    def costo_efectivo(self, camino):
        """``(C efectivo, C valuado)`` de un conjunto de PEs. El valuado es ``valuar_camino``; el
        efectivo es ese mismo C si los PEs cumplen los tipos de PE (``cumple_tipos``) y ``P_fallo``
        si no, como en el contrato (una configuración que no es válida cuesta P_fallo)."""
        c = self.valuar_camino(camino)["C"]
        valido = self.cumple_tipos({self.pesos[pe].tipo for pe in camino})
        return (c if valido else self.costos.p_fallo), c

    def caminar(self, filas, columnas, semilla=None):
        """Desde ``inicio``, epsilon-greedy entre los vecinos sin visitar. En cada paso, con
        probabilidad ``epsilon`` decide el agente RL (o, sin agente RL, uno de los libres al azar), y
        si no, greedy: el vecino que deja el menor C efectivo (``costo_efectivo``; en empate, el de
        menor C valuado y luego el primero fila por fila). En el punto de paro (el camino cumple los
        tipos de PE y ningún vecino libre baja su C), con probabilidad ``epsilon`` el agente RL (o el
        azar) decide parar o seguir ``n`` PEs más (``decidir_extension``; queda en
        ``self.extensiones``); si no, para (``self.parada = "sin_mejora"``). También termina cuando el
        PE actual no tiene vecinos sin visitar (``"sin_vecinos"``). Deja en ``self.decisiones`` qué se hizo en
        cada paso. ``semilla``: la del azar (por defecto la de los parámetros). Necesita ``self.pesos``
        y ``self.evaluacion`` (los pone ``buscar``)."""
        epsilon = self.parametros.epsilon
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon = {epsilon} debe estar entre 0 y 1")
        self.semilla_actual = self.parametros.semilla if semilla is None else semilla
        azar = random.Random(self.semilla_actual)
        f0, c0 = self.parametros.inicio
        if not (0 <= f0 < filas and 0 <= c0 < columnas):
            raise ValueError(f"inicio ({f0}, {c0}) fuera de la malla {filas}x{columnas}")
        ids = Malla(filas, columnas)
        orden = {pe: i for i, pe in enumerate(ids.ids())}
        actual = ids.id_de(f0, c0)
        camino, visitados = [actual], {actual}
        self.decisiones = ["inicio"]
        self.parada = None
        self.extensiones = []
        extra = 0  # pasos que faltan de una extensión (sin revisar el paro)
        while True:
            libres = [v for v in self.vecinos(actual, filas, columnas) if v not in visitados]
            if not libres:
                self.parada = "sin_vecinos"
                return camino
            c_actual = self.costo_efectivo(camino)[0]
            costos = {v: self.costo_efectivo(camino + [v]) for v in libres}
            greedy = min(libres, key=lambda v: (costos[v], orden[v]))
            if extra == 0 and self.cumple_tipos({self.pesos[pe].tipo for pe in camino}) \
                    and costos[greedy][0] >= c_actual:
                # punto de paro: ya cumple los tipos y ningún vecino baja C (según el estimado)
                if azar.random() < epsilon:
                    extra = self.decidir_extension(actual, libres, camino, costos, c_actual, azar, orden)
                if extra == 0:
                    self.parada = "sin_mejora"
                    return camino
            if azar.random() < epsilon:
                actual = azar.choice(sorted(libres, key=orden.get))
                self.decisiones.append("exploracion")
            else:
                actual = greedy
                self.decisiones.append("explotacion")
            camino.append(actual)
            visitados.add(actual)
            if extra > 0:
                extra -= 1

    def decidir_extension(self, actual, libres, camino, costos, c_actual, azar, orden):
        """En el punto de paro: cuántos PEs más sigue el camino (0 = parar), elegido al azar de 0 a
        ``max_extension``. Queda en ``self.extensiones``."""
        maximo = self.parametros.max_extension
        if maximo < 1:
            return 0
        n, decision = azar.randint(0, maximo), "exploracion"
        self.extensiones.append({"paso": len(camino) - 1, "pes": len(camino), "decision": decision, "n": n})
        return n

    def preparar(self, evaluacion=None):
        """Simula la malla de referencia (o usa ``evaluacion`` si ya se tiene) y calcula los pesos."""
        malla = self.malla()
        self.evaluacion = evaluacion if evaluacion is not None else self.medir(self.configuracion(malla))
        if not self.evaluacion.valida:
            raise ValueError(f"la malla {len(malla)}x{len(malla[0])} no es correcta "
                             f"({self.evaluacion.estado}): {self.evaluacion.motivo}")
        self.pesos = self.calcular_pesos(self.evaluacion, malla)
        return malla

    def buscar(self, evaluacion=None, semilla=None):
        """Simula la malla (o usa ``evaluacion`` si ya se tiene), calcula los pesos y recorre el camino
        con ``semilla`` (por defecto la de los parámetros). Devuelve la lista de PEs del camino."""
        malla = self.preparar(evaluacion)
        self.camino = self.caminar(len(malla), len(malla[0]), semilla)
        return self.camino

    def descender(self, evaluacion=None, semilla=None):
        """Modo ``bloques``: un episodio de descenso por bloques desde la malla de referencia
        (``dijkstra/bloques.py``). Devuelve el episodio."""
        from .bloques import descender
        self.preparar(evaluacion)
        self.semilla_actual = self.parametros.semilla if semilla is None else semilla
        self.camino, self.decisiones, self.parada, self.extensiones = [], [], None, []
        self.episodio = descender(self, random.Random(self.semilla_actual))
        self.decisiones, self.parada = list(self.episodio["decisiones"]), self.episodio["parada"]
        return self.episodio

    # -- episodio: el rectángulo del camino, simulado --------------------------------------
    def letra_memoria(self):
        """La letra del tipo de PE de memoria del esquema (la de los bordes oeste y este)."""
        letras = [letra for letra, t in self.base["tipos_pe"].items()
                  if {i.upper() for i in t["instrucciones"]} & MEMORIA_OPS
                  and not {i.upper() for i in t["instrucciones"]} & COMPUTO_OPS]
        if len(letras) != 1:
            raise ValueError(f"se esperaba un solo tipo de PE de memoria en tipos_pe, hay {letras}")
        return letras[0]

    def malla_episodio(self, camino=None):
        """La malla simulable del camino: el rectángulo de filas y columnas que ocupa en la malla,
        sin las columnas que en ese rectángulo son solo de memoria, con una columna de memoria al
        oeste y otra al este (la regla de bordes de costos.json). Los PEs del rectángulo que no están
        en el camino quedan con su tipo de la malla."""
        camino = self.camino if camino is None else camino
        if not camino:
            raise ValueError("no hay camino para formar el episodio")
        malla = self.malla()
        m = self.letra_memoria()
        posiciones = [Malla.posicion(pe) for pe in camino]
        f0, f1 = min(f for f, _ in posiciones), max(f for f, _ in posiciones)
        c0, c1 = min(c for _, c in posiciones), max(c for _, c in posiciones)
        interiores = [c for c in range(c0, c1 + 1) if any(malla[f][c] != m for f in range(f0, f1 + 1))]
        return tuple((m,) + tuple(malla[f][c] for c in interiores) + (m,) for f in range(f0, f1 + 1))

    @staticmethod
    def _clave(configuracion):
        return json.dumps(configuracion, sort_keys=True)

    def simular_episodio(self, configuracion):
        """``(Evaluacion, repetido)`` del rectángulo. Si esa configuración ya está en
        ``cache_episodios`` (de esta corrida o de una anterior, con --continuar) no se vuelve a simular;
        si no, se simula con ``medir`` y se guarda. ``repetido`` dice si el rectángulo ya había salido
        en esta corrida."""
        clave = self._clave(configuracion)
        repetido = clave in self._vistos
        self._vistos.add(clave)
        if clave in self.cache_episodios:
            return self.cache_episodios[clave], repetido
        e = self.medir(configuracion)
        if e.estado not in NO_ATRIBUIBLES:  # un fallo de infraestructura se puede reintentar
            self.cache_episodios[clave] = e
        if self.al_simular:
            self.al_simular(configuracion, e)
        return e, repetido

    def evaluar_episodio(self, evaluacion=None, repetido=False):
        """Simula la malla del episodio (``malla_episodio``) con ``simular_episodio`` (o usa
        ``evaluacion`` si ya se tiene) y calcula la recompensa del episodio ``R = C_best_prev − C``,
        con ``C_best_prev`` = ``self.c_best`` (el mejor C real hasta ahora; si no hay, el de la malla
        completa); ``−P_fallo`` si no es correcta. Devuelve ``self.episodio``."""
        if self.evaluacion is None or not self.camino:
            raise ValueError("primero hay que recorrer el camino (buscar)")
        malla = self.malla_episodio()
        if evaluacion is None:
            evaluacion, repetido = self.simular_episodio(self.configuracion(malla))
        e = evaluacion
        self.evaluacion_episodio = e
        mejor_previo = self.c_best if self.c_best is not None else self.evaluacion.costo
        m = e.metricas or {}
        self.episodio = {
            "n": self.n_episodio,
            "semilla": self.semilla_actual,
            "camino": list(self.camino),
            "decisiones": list(self.decisiones),
            "parada": self.parada,
            "extensiones": list(self.extensiones),
            "extension": sum(x["n"] for x in self.extensiones),
            "repetido": repetido,
            "malla": [" ".join(fila) for fila in malla],
            "filas": len(malla),
            "columnas": len(malla[0]),
            "pes": len(malla) * len(malla[0]),
            "estado": e.estado,
            "motivo": e.motivo,
            "C": e.costo,
            "C_A": m.get("C_A"),
            "C_A_norm": m.get("C_A_norm"),
            "C_L_norm": m.get("C_L_norm"),
            "C_R": m.get("C_R"),
            "costo_por_operacion": {op: o.costo for op, o in e.operaciones.items()},
            "C_estimado": self.valuar_camino()["C"],
            "C_best_prev": mejor_previo,
            "recompensa": recompensa(e.estado, e.costo, mejor_previo, self.costos.p_fallo),
            "formula": "R = C_best_prev − C (correcta); −P_fallo (penalizada); C_best_prev = el mejor C real "
                       "de los episodios anteriores (al principio, el de la malla completa). C es el de la "
                       "simulación del rectángulo del camino; C_estimado, el de la valuación sin volver a "
                       "simular que usa el greedy en cada paso",
        }
        return self.episodio

    def correr(self, evaluacion=None, al_episodio=None):
        """El ciclo de episodios (retroalimentación agente ↔ evaluador). Cada episodio recorre un camino
        con la semilla ``semilla + n``, simula su rectángulo y calcula su C real y su recompensa. Si el C
        real es menor que el mejor (más allá de ``tolerancia_empate``), ese episodio pasa a ser el mejor;
        si no, se descarta y cuenta para la paciencia (un fallo de infraestructura no cuenta). Termina
        con ``max_episodios`` o al llegar a ``paciencia`` episodios seguidos sin mejora. Al final, el
        camino del agente (``camino``, ``decisiones``, ``episodio``...) es el del mejor episodio (o el del
        último, si ninguno fue correcto). ``al_episodio(episodio)`` se llama tras cada episodio.
        Devuelve el historial."""
        if self.parametros.max_episodios < 1 or self.parametros.paciencia < 1:
            raise ValueError("max_episodios y paciencia deben ser al menos 1")
        if self.parametros.modo not in ("camino", "bloques"):
            raise ValueError(f"modo = {self.parametros.modo!r}: use 'camino' o 'bloques'")
        self.historial, self.mejor, self.motivo_fin = [], None, None
        self.c_best, self._vistos = None, set()
        sin_mejora = 0
        for n in range(self.parametros.max_episodios):
            self.n_episodio = n
            inicio_episodio = time.monotonic()
            if self.parametros.modo == "bloques":
                if self.c_best is None:
                    self.preparar(evaluacion)
                    self.c_best = self.evaluacion.costo  # C_best,0: la malla completa (punto de partida)
                ep = self.descender(self.evaluacion, semilla=self.parametros.semilla + n)
            else:
                self.buscar(evaluacion if n == 0 else self.evaluacion, semilla=self.parametros.semilla + n)
                if self.c_best is None:
                    self.c_best = self.evaluacion.costo  # C_best,0: la malla completa (punto de partida)
                ep = self.evaluar_episodio()
            mejora = (ep["estado"] == CORRECTA and ep["C"] is not None
                      and ep["C"] < self.c_best - self.costos.tolerancia_empate)
            ep["mejora"] = mejora
            ep["duracion_s"] = round(time.monotonic() - inicio_episodio, 2)
            if mejora:
                self.c_best, self.mejor = ep["C"], ep
                sin_mejora = 0
            elif ep["estado"] not in NO_ATRIBUIBLES:
                sin_mejora += 1
            ep["C_best"] = self.c_best
            self.historial.append(ep)
            if al_episodio:
                al_episodio(ep)
            if sin_mejora >= self.parametros.paciencia:
                self.motivo_fin = "paciencia"
                break
        else:
            self.motivo_fin = "max_episodios"
        elegido = self.mejor or self.historial[-1]
        self.camino, self.decisiones = list(elegido["camino"]), list(elegido["decisiones"])
        self.parada, self.semilla_actual, self.n_episodio = elegido["parada"], elegido["semilla"], elegido["n"]
        self.extensiones = list(elegido["extensiones"])
        self.episodio = elegido
        return self.historial

    # -- valuación del camino -----------------------------------------------------------
    def valuar_camino(self, camino=None):
        """Costo de los PEs del camino con la función de costo (contrato.py), sin volver a simular:
        los ciclos, los NOPs de cada PE y C_R salen de la evaluación de la malla. ``camino``: los
        PEs a valuar (por defecto ``self.camino``; ``historia`` valúa cada prefijo).

        C_A = Σ w_tipo de los PEs del camino; por operación, C_L = latencia(ciclos, NOPs del
        camino, PEs del camino); C = promedio de K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R."""
        camino = self.camino if camino is None else camino
        n = self.evaluacion.normalizadores
        if not n:
            raise ValueError("la evaluación de la malla no trae normalizadores")
        k = self.costos.K
        pes = len(camino)
        c_a = sum(self.pesos[pe].w_tipo for pe in camino)
        por_operacion = {}
        for op, e in self.evaluacion.operaciones.items():
            nops = sum(e.nops_por_pe[pe] for pe in camino)
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

    # -- historia del camino ----------------------------------------------------------
    def cumple_tipos(self, tipos):
        """Si un conjunto de tipos de PE cumple las reglas de tipos de costos.json: al menos un PE de
        memoria y uno de cómputo (``requiere_memoria_y_computo``) y al menos un escalar (E) y un
        vectorial (V) (``requiere_escalar_y_vectorial``). Las reglas de bordes no aplican: el camino
        no es un rectángulo."""
        r = self.costos.restricciones
        por_tipo = [{i.upper() for i in self.base["tipos_pe"][t]["instrucciones"]} for t in tipos]
        todas = set().union(*por_tipo)
        if r.get("requiere_memoria_y_computo") and not (todas & MEMORIA_OPS and todas & COMPUTO_OPS):
            return False
        if r.get("requiere_escalar_y_vectorial") and not (
                any(ops & ESCALAR_OPS for ops in por_tipo) and any(ops & VECTORIAL_OPS for ops in por_tipo)):
            return False
        return True

    def historia(self):
        """El camino paso a paso, para graficarlo después: en cada paso, el PE que se agregó, quién lo
        eligió y la valuación del prefijo (los PEs del camino hasta ese paso) con ``valuar_camino``,
        sin volver a simular. ``cumple_tipos`` dice si el prefijo cumple las reglas de tipos de PE;
        ``mejor`` es el paso de menor C entre los que las cumplen (solo informativo: el resultado
        sigue siendo el camino completo). ``C_efectivo`` es el C que usa el greedy (``P_fallo`` si el
        prefijo no cumple los tipos) y ``recompensa`` = C_efectivo del paso anterior − C_efectivo de
        este paso (positiva si el paso bajó el costo; ``None`` en el inicio)."""
        malla = self.malla()
        pasos, tipos = [], set()
        anterior = None
        for i, (pe, decision) in enumerate(zip(self.camino, self.decisiones)):
            p = self.pesos[pe]
            tipos.add(p.tipo)
            v = self.valuar_camino(self.camino[:i + 1])
            efectivo = self.costo_efectivo(self.camino[:i + 1])[0]
            pasos.append({
                "paso": i, "pe": pe, "fila": Malla.posicion(pe)[0], "columna": Malla.posicion(pe)[1],
                "tipo": p.tipo, "peso": p.peso, "decision": decision, "pes": i + 1,
                "C": v["C"], "C_A": v["C_A"], "C_A_norm": v["C_A_norm"], "C_L_norm": v["C_L_norm"],
                "C_R": v["C_R"], "costo_por_operacion": {op: o["costo"] for op, o in v["por_operacion"].items()},
                "cumple_tipos": self.cumple_tipos(tipos),
                "C_efectivo": efectivo, "recompensa": None if anterior is None else anterior - efectivo,
            })
            anterior = efectivo
        validos = [p for p in pasos if p["cumple_tipos"]]
        mejor = min(validos, key=lambda p: (p["C"], p["paso"])) if validos else None
        return {
            "malla": [" ".join(fila) for fila in malla],
            "filas": len(malla),
            "columnas": len(malla[0]),
            "C_malla": self.evaluacion.costo,
            "epsilon": self.parametros.epsilon,
            "semilla": self.semilla_actual,
            "parada": self.parada,
            "extensiones": self.extensiones,
            "mejor": None if mejor is None else {k: mejor[k] for k in ("paso", "pe", "pes", "C", "C_A", "C_L_norm")},
            "pasos": pasos,
            "nota": "Cada paso valúa el prefijo del camino con los ciclos, NOPs y C_R de la simulación de la "
                    "malla completa (sin volver a simular). cumple_tipos: reglas de tipos de PE de costos.json "
                    "(memoria y cómputo, E y V); no las de bordes. C_efectivo: C si cumple los tipos, si no "
                    "P_fallo. recompensa: C_efectivo anterior − C_efectivo (positiva si bajó el costo). "
                    "decision: inicio, explotacion (greedy: el vecino de menor C efectivo) o "
                    "exploracion (al azar). "
                    "parada: sin_mejora (cumple los tipos y ningún vecino baja C) o sin_vecinos.",
        }

    # -- resultado --------------------------------------------------------------------
    def resultado(self):
        """Los PEs del camino con su tipo y peso, y el mapa de la malla con '.' en los que quedan fuera.
        En el modo ``bloques``: la malla final del episodio elegido y su descenso."""
        if self.parametros.modo == "bloques":
            ep = self.episodio or {}
            return {
                "modo": "bloques", "malla": ep.get("malla"), "filas": ep.get("filas"),
                "columnas": ep.get("columnas"), "pes": ep.get("pes"), "C": ep.get("C"),
                "epsilon": self.parametros.epsilon, "semilla": self.semilla_actual,
                "bloques": list(self.parametros.bloques), "parada": self.parada,
                "exploraciones": self.decisiones.count("exploracion"), "episodio": self.episodio,
            }
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
            "semilla": self.semilla_actual,
            "exploraciones": self.decisiones.count("exploracion"),
            "parada": self.parada,
            "extensiones": self.extensiones,
            "episodio": self.episodio,
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
    operacion, carpeta, ruta_costos, t_max, configuracion, mapeador_procesos = argumentos
    evaluador = Evaluador([operacion], carpeta, _costos(ruta_costos, t_max),
                          opciones=Opciones(procesos=mapeador_procesos))
    return evaluador.evaluar(configuracion).operaciones[operacion]


def medir_en_paralelo(configuracion, operaciones, carpeta, ruta_costos, t_max=None, procesos=None,
                      mapeador_procesos=1):
    """Evalúa la configuración con cada operación en su propio proceso y junta los costos
    como ``Evaluador.evaluar`` (mismos normalizadores y ecuaciones). ``mapeador_procesos``
    reparte la búsqueda del mapeador entre procesos sin cambiar su resultado."""
    from .evaluador import EvaluacionOperacion, Evaluador, agregar
    costos = _costos(ruta_costos, t_max)
    evaluador = Evaluador(list(operaciones), carpeta, costos, opciones=Opciones(procesos=mapeador_procesos))
    if evaluador.motivos_invalida(configuracion):
        return evaluador.evaluar(configuracion)
    argumentos = [(op, str(carpeta), str(ruta_costos), t_max, configuracion, mapeador_procesos)
                  for op in operaciones]
    with ProcessPoolExecutor(procesos or len(operaciones)) as pool:
        por_op = dict(zip(operaciones, pool.map(_evaluar_operacion, argumentos)))
    por_op = {op: EvaluacionOperacion(**asdict(e)) for op, e in por_op.items()}
    simuladas = any(e.C_L is not None for e in por_op.values())
    normalizadores = evaluador.normalizadores() if simuladas else evaluador._fijos
    return agregar(Evaluador.formato(configuracion), por_op, costos, normalizadores)
