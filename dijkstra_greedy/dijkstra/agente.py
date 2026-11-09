"""Agente Greedy (solución Dijkstra + Greedy): busca la configuración de la
CGRA de menor costo y guarda la cadena de mejoras.

Es la caja "Agente" del ciclo de retroalimentación del PDF: propone
configuraciones (tamaño, tipos de PE y su distribución), el evaluador
(``evaluador.py``: Dijkstra + simulador) devuelve sus costos y el agente
compara ``C = K_A·C_A + K_L·C_L + K_R·C_R``. Guarda una **cadena de mejoras**:
cada configuración nueva se compara con la última de la cadena; si su C es
mayor, se descarta (penalización); si es menor o igual, se agrega a la cadena
(recompensa) y las siguientes se comparan con ella. La última es la mejor.

Contrato con la solución RL (``solucion_rl.md``, sección 10): la ronda 0 es
la referencia ``M E M / M V M / M E M`` (``C_best,0 = C_ref``), solo se
proponen configuraciones que pasan la misma validación previa que el RL
(``contrato.validar_accion``) y la búsqueda termina con el mismo presupuesto:
``max_evaluaciones`` (sin contar la referencia) y ``paciencia`` (evaluaciones
seguidas sin mejorar el mínimo en más de ``tolerancia_empate_costo``).
Después, en cada ronda se exploran las vecinas (a un cambio de distancia) de
las mejores de la cadena.

Opcional (``barrido``, apagado por defecto porque no empieza en la referencia):
la ronda 0 barre todos los tamaños de menos a más PEs, con cotas inferiores de
C por área para descartar lo que no puede entrar a la cadena.

El costo lo calcula solo el evaluador; el agente nada más compara
``evaluacion.costo``. Para probarlo sin el pipeline, ``medir`` puede ser
cualquier función ``configuración (dict de cgra.json) -> Evaluacion``.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import leer_cgra, rango
from formato_cgra import FormatoCGRA, generar_malla  # noqa: E402 (cgra/src/ del simulador)

from .contrato import CORRECTA, NO_ATRIBUIBLES, Costos, validar_accion

RUTA_PARAMETROS = Path(__file__).resolve().parents[1] / "config" / "agente.json"
ARITMETICA = frozenset({"ADD", "SUB", "MUL", "DIV", "VADD", "VSUB", "VMUL", "VDIV"})


@dataclass
class Parametros:
    max_rondas: int = 50
    lado_minimo: int = None
    lado_maximo: int = None
    barrido: bool = False
    lote_barrido: int = 16
    iniciales: object = "referencia"  # "referencia" o lista de tamaños [filas, columnas]
    interior_alternativo: str = "C"
    procesos: int = None
    max_evaluaciones: int = 10
    paciencia: int = 3

    @classmethod
    def cargar(cls, ruta=RUTA_PARAMETROS):
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**{clave: datos[clave] for clave in cls.__dataclass_fields__ if clave in datos})


@dataclass
class Registro:
    """Una configuración evaluada (o descartada antes de evaluar)."""

    malla: tuple  # tupla de filas, cada una una tupla de letras
    clave: str
    ronda: int
    evaluacion: object = None  # Evaluacion del evaluador, o None si se descartó antes
    motivo: str = ""

    @property
    def costo(self):
        if self.evaluacion is None or self.evaluacion.costo is None:
            return float("inf")
        return self.evaluacion.costo

    @property
    def area(self):
        if self.evaluacion is None or self.evaluacion.area is None:
            return float("inf")
        return self.evaluacion.area

    @property
    def estado(self):
        return self.evaluacion.estado if self.evaluacion is not None else None

    @property
    def valida(self):
        return self.evaluacion is not None and self.evaluacion.valida

    def orden(self):
        return (self.costo, self.area, self.clave)

    def como_dict(self):
        datos = {"ronda": self.ronda, "clave": self.clave, "malla": [" ".join(f) for f in self.malla],
                 "costo": self.costo, "valida": self.valida, "estado": self.estado, "motivo": self.motivo}
        if self.evaluacion is not None:
            e = self.evaluacion
            datos.update(
                costo=e.costo, metricas={k: v for k, v in e.metricas.items() if k != "capacidades"},
                operaciones={op: {"estado": o.estado, "modo": o.modo, "ciclos": o.ciclos, "C_L": o.C_L,
                                  "C_R": o.C_R, "costo": o.costo, "motivo": o.motivo}
                             for op, o in e.operaciones.items()},
                motivo=self.motivo or e.motivo,
            )
        return datos


def clave(malla):
    """``3x3:M,E,M|M,V,M|M,C,M``"""
    return f"{len(malla)}x{len(malla[0])}:" + "|".join(",".join(fila) for fila in malla)


class Agente:
    """Búsqueda determinística de configuraciones de la CGRA con una cadena de mejoras."""

    def __init__(self, medir, base=None, parametros=None, medir_lote=None, al_evaluar=None, cota=None,
                 cota_tamano=None, costos=None, validar=None):
        """``medir(configuración) -> Evaluacion``; ``medir_lote(lista) -> lista`` evalúa varias
        a la vez (en paralelo) y debe devolverlas en el mismo orden. ``base``: el
        dict de cgra.json (tipos, distribución, rangos); ``al_evaluar(registro)``
        se llama con cada configuración nueva (para el historial).

        ``cota(configuración)`` y ``cota_tamano(tipos_pe, pes)``: cotas inferiores
        de C del evaluador (``Evaluador.cota`` y ``Evaluador.cota_tamano``). Sin
        ellas no se descarta nada por costo.

        ``costos``: parámetros del contrato (costos.json: referencia, restricciones,
        tolerancia de empate); ``validar(configuración) -> [motivos]``: validación
        previa (por defecto ``contrato.validar_accion`` con el esquema de ``base``)."""
        self.medir = medir
        self.cota = cota
        self.cota_tamano = cota_tamano
        self.barrido = {"tamanos": 0, "revisados": 0, "hasta": None, "sin_revisar": 0}
        self.medir_lote = medir_lote or (lambda configuraciones: [medir(c) for c in configuraciones])
        self.base = base if base is not None else leer_cgra()
        self.parametros = parametros or Parametros()
        self.al_evaluar = al_evaluar
        self.costos = costos or Costos.cargar()
        self.validar = validar or self._validar
        self.tipos = list(self.base["tipos_pe"])
        instrucciones = {letra: {i.upper() for i in t["instrucciones"]} for letra, t in self.base["tipos_pe"].items()}
        self.de_memoria = {t for t in self.tipos if "LD" in instrucciones[t]}
        self.de_computo = {t for t in self.tipos if instrucciones[t] & ARITMETICA}
        # Solo memoria (M): no se cambian en los movimientos por fila.
        self.solo_memoria = [t for t in self.tipos if t in self.de_memoria and t not in self.de_computo]
        self.solo_computo = [t for t in self.tipos if t in self.de_computo and t not in self.de_memoria]
        p = self.parametros
        self.limites = []
        for nombre in ("filas", "columnas"):
            minimo, maximo = rango(self.base[nombre])
            if p.lado_minimo is not None:
                minimo = max(minimo, p.lado_minimo)
            if p.lado_maximo is not None:
                maximo = min(maximo, p.lado_maximo)
            self.limites.append((minimo, maximo))
        self.registros = {}  # clave -> Registro
        self.historial = []  # registros en el orden en que se evaluaron
        self.cadena = []  # registros válidos, cada uno de C menor o igual que el anterior
        self.cadena_por_ronda = []
        self.evaluadas = 0  # evaluaciones del presupuesto (sin la referencia)
        self.c_min = None  # menor C correcto (óptimas del RL)
        self.sin_mejora = 0
        self.terminada = None  # motivo de fin por presupuesto

    def _validar(self, configuracion):
        from .evaluador import Evaluador
        try:
            formato = Evaluador.formato(configuracion)
        except (ValueError, TypeError, KeyError) as error:
            return [f"cgra.json: {error}"]
        return validar_accion(configuracion, formato, self.costos.restricciones, self.base["tipos_pe"])

    def referencia(self):
        return tuple(tuple(fila.split()) for fila in self.costos.referencia)

    # -- configuraciones --------------------------------------------------------
    def configuracion(self, malla):
        """La configuración que recibe el evaluador: cgra.json con tamaño fijo y ``malla`` explícita."""
        datos = {k: v for k, v in self.base.items() if k not in ("distribucion", "malla", "descripcion")}
        datos.update(filas=len(malla), columnas=len(malla[0]), malla=[" ".join(fila) for fila in malla])
        return datos

    def iniciales(self):
        """Las configuraciones de partida (sin barrido): la referencia, o los tamaños de ``iniciales``."""
        if self.parametros.iniciales == "referencia":
            return [self.referencia()]
        mallas = []
        for filas, columnas in self.parametros.iniciales:
            filas = min(max(filas, self.limites[0][0]), self.limites[0][1])
            columnas = min(max(columnas, self.limites[1][0]), self.limites[1][1])
            mallas.extend(self.de_distribucion(filas, columnas))
        return list(dict.fromkeys(mallas))

    def de_distribucion(self, filas, columnas):
        """La malla de la distribución de cgra.json en ese tamaño y su variante con el interior alternativo."""
        if "distribucion" not in self.base:
            raise ValueError("las configuraciones iniciales salen de 'distribucion' de cgra.json, y no la tiene")
        malla = tuple(tuple(fila) for fila in generar_malla(filas, columnas, self.base["distribucion"]))
        mallas = [malla]
        alternativo = self.parametros.interior_alternativo
        if alternativo:
            mallas.append(tuple(tuple(t if t in self.solo_memoria else alternativo for t in fila) for fila in malla))
        return list(dict.fromkeys(mallas))

    def tamanos(self):
        """Todos los tamaños de los límites, de menos a más PEs (y por filas, columnas)."""
        (fmin, fmax), (cmin, cmax) = self.limites
        return sorted(((f, c) for f in range(fmin, fmax + 1) for c in range(cmin, cmax + 1)),
                      key=lambda t: (t[0] * t[1], t))

    def motivo_invalida(self, malla):
        """Validación previa barata; ``None`` si se puede evaluar."""
        filas, columnas = len(malla), len(malla[0])
        (fmin, fmax), (cmin, cmax) = self.limites
        if not (fmin <= filas <= fmax and cmin <= columnas <= cmax):
            return f"tamaño {filas}x{columnas} fuera de los límites (filas {fmin}..{fmax}, columnas {cmin}..{cmax})"
        if any(len(fila) != columnas for fila in malla):
            return "las filas de la malla no tienen el mismo largo"
        motivos = self.validar(self.configuracion(malla))
        return motivos[0] if motivos else None

    def vecinas(self, malla):
        """Configuraciones a un cambio de distancia, siempre en el mismo orden."""
        filas, columnas = len(malla), len(malla[0])
        lista = [list(fila) for fila in malla]
        resultado = []

        def agregar(nueva):
            resultado.append(tuple(tuple(fila) for fila in nueva))

        # 1. Movimiento grande: todos los PEs que no son solo de memoria de una fila, a un tipo de cómputo.
        for f in range(filas):
            for tipo in self.solo_computo:
                nueva = [fila[:] for fila in lista]
                nueva[f] = [t if t in self.solo_memoria else tipo for t in nueva[f]]
                agregar(nueva)
        # 2. Agregar una fila o columna: copia de la interior más cercana al borde sur / este.
        origen = filas - 2 if filas >= 3 else filas - 1
        agregar(lista[:filas - 1] + [lista[origen][:]] + lista[filas - 1:] if filas >= 2 else lista + [lista[0][:]])
        origen = columnas - 2 if columnas >= 3 else columnas - 1
        if columnas >= 2:
            agregar([fila[:columnas - 1] + [fila[origen]] + fila[columnas - 1:] for fila in lista])
        else:
            agregar([fila + [fila[0]] for fila in lista])
        # 3. Quitar cada fila y cada columna.
        if filas > 1:
            for f in range(filas):
                agregar(lista[:f] + lista[f + 1:])
        if columnas > 1:
            for c in range(columnas):
                agregar([fila[:c] + fila[c + 1:] for fila in lista])
        # 4. Cambiar el tipo de un PE.
        for f in range(filas):
            for c in range(columnas):
                for tipo in self.tipos:
                    if tipo != lista[f][c]:
                        nueva = [fila[:] for fila in lista]
                        nueva[f][c] = tipo
                        agregar(nueva)
        # 5. Mover un PE de memoria al siguiente lugar del perímetro (sentido horario).
        perimetro = self._perimetro(filas, columnas)
        for i, (f, c) in enumerate(perimetro):
            if lista[f][c] in self.solo_memoria and len(perimetro) > 1:
                f2, c2 = perimetro[(i + 1) % len(perimetro)]
                if lista[f2][c2] != lista[f][c]:
                    nueva = [fila[:] for fila in lista]
                    nueva[f][c], nueva[f2][c2] = nueva[f2][c2], nueva[f][c]
                    agregar(nueva)
        return [v for v in dict.fromkeys(resultado) if v != malla]

    @staticmethod
    def _perimetro(filas, columnas):
        """Celdas del borde en sentido horario desde (0, 0)."""
        if filas == 1:
            return [(0, c) for c in range(columnas)]
        if columnas == 1:
            return [(f, 0) for f in range(filas)]
        arriba = [(0, c) for c in range(columnas)]
        derecha = [(f, columnas - 1) for f in range(1, filas)]
        abajo = [(filas - 1, c) for c in range(columnas - 2, -1, -1)]
        izquierda = [(f, 0) for f in range(filas - 2, 0, -1)]
        return arriba + derecha + abajo + izquierda

    # -- evaluación ---------------------------------------------------------------
    def evaluar(self, mallas, ronda, umbral=float("inf")):
        """Evalúa las mallas que todavía no se evaluaron; devuelve sus registros, en orden.
        Las que según ``cota`` no pueden bajar de ``umbral`` se descartan sin medir."""
        nuevas = [m for m in dict.fromkeys(mallas) if clave(m) not in self.registros]
        if self.terminada:
            nuevas = []
        por_medir = []
        for malla in nuevas:
            registro = Registro(malla, clave(malla), ronda)
            registro.motivo = self.motivo_invalida(malla) or ""
            if not registro.motivo and self.cota and umbral < float("inf"):
                cota = self.cota(self.configuracion(malla))
                if cota > umbral:
                    registro.motivo = f"no puede entrar a la cadena: C ≥ {cota:.6g} > {umbral:.6g}"
            self.registros[registro.clave] = registro
            if not registro.motivo:
                por_medir.append(registro)
        # Presupuesto: no se miden más de las que quedan (la referencia no cuenta).
        restantes = self._restantes()
        if restantes is not None:
            sobran = [r for r in por_medir if r.malla != self.referencia()][restantes:]
            for registro in sobran:
                del self.registros[registro.clave]
            por_medir = [r for r in por_medir if r not in sobran]
            nuevas = [m for m in nuevas if clave(m) in self.registros]
        for registro, evaluacion in zip(por_medir,
                                        self.medir_lote([self.configuracion(r.malla) for r in por_medir])):
            registro.evaluacion = evaluacion
        for malla in nuevas:
            registro = self.registros[clave(malla)]
            if registro.evaluacion is not None and self.terminada:
                del self.registros[registro.clave]  # se midió en el lote, pero ya no cabe
                continue
            self.historial.append(registro)
            if registro.evaluacion is not None:
                self._contar(registro)
            if self.al_evaluar:
                self.al_evaluar(registro)
        return [self.registros[clave(m)] for m in mallas if clave(m) in self.registros]

    def _restantes(self):
        maximo = self.parametros.max_evaluaciones
        return None if maximo is None else max(maximo - self.evaluadas, 0)

    def _contar(self, registro):
        """Presupuesto y paciencia, en el orden de evaluación, con la regla de óptimas del RL."""
        es_referencia = registro.malla == self.referencia()
        e = registro.evaluacion
        mejora = False
        if e.estado == CORRECTA and e.costo is not None:
            if self.c_min is None or e.costo < self.c_min - self.costos.tolerancia_empate:
                self.c_min, mejora = e.costo, True
        if not es_referencia:
            self.evaluadas += 1
            if e.estado not in NO_ATRIBUIBLES:
                self.sin_mejora = 0 if mejora else self.sin_mejora + 1
        p = self.parametros
        if p.max_evaluaciones is not None and self.evaluadas >= p.max_evaluaciones:
            self.terminada = "max_evaluaciones"
        elif p.paciencia is not None and self.sin_mejora >= p.paciencia:
            self.terminada = "paciencia"

    def umbral(self):
        """C de la última de la cadena: lo que no pueda bajar de ahí no entra."""
        return self.cadena[-1].costo if self.cadena else float("inf")

    def actualizar_cadena(self, registros):
        """Compara cada registro válido, en orden, con la última de la cadena: si su C es
        mayor se descarta; si es menor o igual, se agrega y los siguientes se comparan
        con él. Devuelve los que entraron."""
        entraron = []
        for registro in registros:
            if not registro.valida or registro.costo > self.umbral():
                continue
            if any(otro.clave == registro.clave for otro in self.cadena):
                continue
            self.cadena.append(registro)
            entraron.append(registro)
        return entraron

    def barrer(self, al_avanzar=None):
        """Ronda 0: todos los tamaños, de menos a más PEs, de a ``lote_barrido``.
        Termina en el primer tamaño en que ninguna malla podría bajar del umbral."""
        p = self.parametros
        tamanos = self.tamanos()
        lote_barrido = max(p.lote_barrido, 1)
        self.barrido = {"tamanos": len(tamanos), "revisados": 0, "hasta": None, "sin_revisar": 0}
        for inicio in range(0, len(tamanos), lote_barrido):
            umbral = self.umbral()
            lote, cortado = [], False
            for filas, columnas in tamanos[inicio:inicio + lote_barrido]:
                if self.cota_tamano and self.cota_tamano(self.base["tipos_pe"], filas * columnas) > umbral:
                    cortado = True
                    break
                lote.extend(self.de_distribucion(filas, columnas))
                self.barrido["revisados"] += 1
                self.barrido["hasta"] = (filas, columnas)
            self.actualizar_cadena(self.evaluar(lote, 0, umbral))
            if al_avanzar:
                al_avanzar(self.barrido, self.cadena)
            if cortado:
                break
        self.barrido["sin_revisar"] = len(tamanos) - self.barrido["revisados"]

    def buscar(self, al_terminar_ronda=None, al_avanzar_barrido=None):
        """Arma la cadena de mejoras y la devuelve en el orden en que se encontró.

        En cada ronda se exploran las vecinas de las mejores de la cadena (la última
        y las que empatan con ella) que todavía no se exploraron. Termina cuando no
        queda ninguna por explorar (ninguna vecina bajó el costo) o a las ``max_rondas``."""
        p = self.parametros
        self.cadena = []
        self.evaluadas, self.c_min, self.sin_mejora, self.terminada = 0, None, 0, None
        if p.barrido:
            self.barrer(al_avanzar_barrido)
        else:
            self.actualizar_cadena(self.evaluar(self.iniciales(), 0))
            referencia = self.registros.get(clave(self.referencia()))
            if p.iniciales == "referencia" and (referencia is None or not referencia.valida):
                self.terminada = "referencia no válida"  # sin referencia no hay C_best,0 (como el RL)
        self.cadena_por_ronda = [list(self.cadena)]
        if al_terminar_ronda:
            al_terminar_ronda(0, self.cadena)
        exploradas = set()
        for ronda in range(1, p.max_rondas + 1):
            if self.terminada:
                break
            por_explorar = [r for r in self.cadena if r.costo == self.umbral() and r.clave not in exploradas]
            if not por_explorar:
                break
            exploradas.update(r.clave for r in por_explorar)
            vecinas = [v for r in por_explorar for v in self.vecinas(r.malla) if clave(v) not in self.registros]
            self.actualizar_cadena(self.evaluar(vecinas, ronda, self.umbral()))
            self.cadena_por_ronda.append(list(self.cadena))
            if al_terminar_ronda:
                al_terminar_ronda(ronda, self.cadena)
        return self.cadena


# -- evaluación en paralelo -----------------------------------------------------
_EVALUADOR = None


def _iniciar_proceso(argumentos):
    global _EVALUADOR
    from .evaluador import Costos, Evaluador
    operaciones, carpeta, ruta_costos, modo = argumentos
    _EVALUADOR = Evaluador(operaciones, carpeta, Costos.cargar(ruta_costos), modo=modo)


def _evaluar_en_proceso(configuracion):
    return _EVALUADOR.evaluar(configuracion)


class EvaluadorParalelo:
    """Evalúa listas de configuraciones repartidas en procesos, cada uno con su
    ``Evaluador``. Devuelve los resultados en el orden de la lista, así que el
    agente decide igual que en serie."""

    def __init__(self, operaciones, carpeta, ruta_costos, modo="auto", procesos=None):
        self.procesos = procesos or os.cpu_count() or 1
        self.argumentos = (operaciones, str(carpeta), str(ruta_costos), modo)
        self._pool = None

    def lote(self, configuraciones):
        if not configuraciones:
            return []
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self.procesos, initializer=_iniciar_proceso,
                                             initargs=(self.argumentos,))
        return list(self._pool.map(_evaluar_en_proceso, configuraciones))

    def cerrar(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
