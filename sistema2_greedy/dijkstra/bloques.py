"""Modo ``bloques``: exploración que reduce la malla de referencia por bloques.

Cada episodio es un **descenso** que parte de la malla de referencia (la 32x32) y la va haciendo más
pequeña. El estado es un rectángulo de la referencia: las filas ``[f0, f1)`` y las columnas interiores
``[k0, k1)`` (las que no son de memoria), con una columna de memoria (M) al oeste y otra al este.

En cada paso, con el tamaño de bloque actual ``b`` (``bloques`` en agente.json: 16, 8, 4, 2, 1), las
acciones son quitar ``b`` filas o ``b`` columnas interiores, del inicio o del final:

    FILAS INICIO · FILAS FIN · COLUMNAS INICIO · COLUMNAS FIN

Greedy elige la acción de menor costo estimado (con los datos de la simulación de la referencia, sin
simular) y, con probabilidad ``epsilon``, se explora: se elige una acción al azar (sin LLM: este
sistema no usa ninguno). La malla elegida **se simula**: si su costo real baja
respecto a la malla actual, se acepta y el descenso sigue desde ella con el mismo bloque; si sube, se
descarta (pág. 5 del PDF) y se prueba otra acción. Cuando ninguna acción del bloque mejora, se pasa al
bloque siguiente; el descenso termina cuando ni el bloque más pequeño mejora.

Fase de tipos (``tipos: true`` en agente.json): al terminar el descenso, sobre la malla que quedó, las
acciones son cambiar el tipo de **un PE del interior** de escalar a vectorial o al revés (E <-> V; los PEs
de memoria de los bordes no se tocan), como el agente de la solución RL, que elige el tipo de cada PE del
interior. Greedy elige el cambio de menor C estimado (el área cambia exactamente; la latencia se supone
igual, así que el estimado favorece pasar de V a E) y, con probabilidad ``epsilon``, se elige un cambio
al azar. Cada malla propuesta se simula y se acepta si su C real baja; la
fase termina cuando ningún cambio mejora la malla actual o al llegar a
``max_cambios_tipo`` simulaciones. La malla resultante ya no es un rectángulo de la referencia: es una
configuración con su propia distribución de tipos.

Todas las mallas simuladas (aceptadas y descartadas) quedan en la caché de la corrida, son puntos del
frente de Pareto de la exploración y no se vuelven a simular en otros episodios.
"""

import time
from dataclasses import dataclass

from malla import Malla  # noqa: E402 (cgra/src/ del simulador)

from .contrato import COMPUTO_OPS, CORRECTA, MEMORIA_OPS, recompensa

ACCIONES = ("FILAS INICIO", "FILAS FIN", "COLUMNAS INICIO", "COLUMNAS FIN")
# Si dos acciones dan la misma malla (en la referencia uniforme, quitar columnas del inicio o del final, o un
# bloque par de filas de arriba o de abajo, deja la misma configuración) se ofrece una sola: la de FIN, para
# que el rectángulo quede anclado en la esquina superior izquierda (PE00) al dibujarlo. La configuración
# simulada es la misma con cualquiera de las dos.
PREFERENCIA = ("FILAS FIN", "FILAS INICIO", "COLUMNAS FIN", "COLUMNAS INICIO")


@dataclass(frozen=True)
class Estado:
    """Rectángulo de la malla de referencia: filas ``[f0, f1)`` y columnas interiores ``[k0, k1)``."""

    f0: int
    f1: int
    k0: int
    k1: int

    @property
    def filas(self):
        return self.f1 - self.f0

    @property
    def interiores(self):
        return self.k1 - self.k0


class Descenso:
    """Las operaciones sobre los rectángulos de la referencia que usa ``Agente.descender``."""

    def __init__(self, agente):
        self.agente = agente
        self.referencia = agente.malla()
        self.memoria = agente.letra_memoria()
        ref, m = self.referencia, self.memoria
        self.interiores = [c for c in range(len(ref[0])) if any(ref[f][c] != m for f in range(len(ref)))]
        bordes = [c for c in range(len(ref[0])) if c not in self.interiores]
        if not self.interiores or not bordes:
            raise ValueError("la malla de referencia necesita columnas de memoria y columnas interiores")
        self.oeste, self.este = min(bordes), max(bordes)
        self.ids = Malla(len(ref), len(ref[0]))
        # los dos tipos de cómputo (sin memoria) que se intercambian en la fase de tipos: {"E": "V", "V": "E"}
        computo = [letra for letra, t in agente.base["tipos_pe"].items()
                   if {i.upper() for i in t["instrucciones"]} & COMPUTO_OPS
                   and not {i.upper() for i in t["instrucciones"]} & MEMORIA_OPS]
        self.alternos = {computo[0]: computo[1], computo[1]: computo[0]} if len(computo) == 2 else {}

    def inicial(self):
        return Estado(0, len(self.referencia), 0, len(self.interiores))

    def malla(self, e):
        ref, m = self.referencia, self.memoria
        return tuple((m,) + tuple(ref[f][self.interiores[k]] for k in range(e.k0, e.k1)) + (m,)
                     for f in range(e.f0, e.f1))

    def pes(self, e):
        """Los PEs de la referencia que representan al rectángulo (para el costo estimado): sus filas,
        sus columnas interiores y las columnas de memoria de la referencia en esas filas."""
        columnas = [self.oeste] + [self.interiores[k] for k in range(e.k0, e.k1)] + [self.este]
        return [self.ids.id_de(f, c) for f in range(e.f0, e.f1) for c in columnas]

    def valido(self, e):
        if e.filas < 1 or e.interiores < 1:
            return False
        return self.agente.cumple_tipos({t for fila in self.malla(e) for t in fila})

    @staticmethod
    def aplicar(e, accion, b):
        """El rectángulo después de quitar ``b`` filas o columnas interiores; ``None`` si no se puede."""
        if accion == "FILAS INICIO":
            nuevo = Estado(e.f0 + b, e.f1, e.k0, e.k1)
        elif accion == "FILAS FIN":
            nuevo = Estado(e.f0, e.f1 - b, e.k0, e.k1)
        elif accion == "COLUMNAS INICIO":
            nuevo = Estado(e.f0, e.f1, e.k0 + b, e.k1)
        elif accion == "COLUMNAS FIN":
            nuevo = Estado(e.f0, e.f1, e.k0, e.k1 - b)
        else:
            raise ValueError(f"acción desconocida: {accion!r}")
        return nuevo if nuevo.filas >= 1 and nuevo.interiores >= 1 else None

    def estimado(self, e):
        return self.agente.costo_efectivo(self.pes(e))[0]

    def rectangulo(self, e):
        """``[fila inicial, fila final, columna inicial, columna final]`` en la referencia (incluye las
        columnas vecinas como bordes), para dibujar el rectángulo sobre la malla de referencia."""
        return [e.f0, e.f1 - 1, max(0, self.interiores[e.k0] - 1),
                min(len(self.referencia[0]) - 1, self.interiores[e.k1 - 1] + 1)]

    def descripcion(self, e):
        malla = self.malla(e)
        return f"{len(malla)}x{len(malla[0])}"


def paso(descenso, estado, evaluacion, accion, bloque, quien, aceptada, nueva, estimado):
    m = evaluacion.metricas or {}
    malla = descenso.malla(estado)
    return {
        "accion": accion, "bloque": bloque, "quien": quien, "aceptada": aceptada, "nueva": nueva,
        "filas": len(malla), "columnas": len(malla[0]), "malla": [" ".join(f) for f in malla],
        "estado_rect": [estado.f0, estado.f1, estado.k0, estado.k1],
        "rectangulo": descenso.rectangulo(estado), "estado": evaluacion.estado, "C": evaluacion.costo,
        "C_estimado": estimado, "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
        "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"),
    }


def paso_tipo(descenso, estado, malla, evaluacion, accion, pe, de, a, quien, aceptada, nueva, estimado):
    """Un paso de la fase de tipos: ``malla`` es la malla propuesta (el PE ``pe`` = [fila, columna] de la
    malla pasa de ``de`` a ``a``); el rectángulo de la referencia es el del final del descenso."""
    m = evaluacion.metricas or {}
    return {
        "accion": accion, "fase": "tipos", "bloque": None, "quien": quien, "aceptada": aceptada, "nueva": nueva,
        "pe": list(pe), "de": de, "a": a, "filas": len(malla), "columnas": len(malla[0]),
        "malla": [" ".join(f) for f in malla], "estado_rect": [estado.f0, estado.f1, estado.k0, estado.k1],
        "rectangulo": descenso.rectangulo(estado), "estado": evaluacion.estado, "C": evaluacion.costo,
        "C_estimado": estimado, "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
        "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"),
    }


def _tiempos(t, t_decision, evaluacion, repetida):
    """Agrega al paso lo que tardó la decisión de Greedy (o del azar), lo que costó evaluar la malla
    (``tiempo_s`` de la evaluación; si vino de la caché, lo que tardó cuando se simuló) y si la malla ya
    se había probado en esta corrida (``repetida``: no se vuelve a evaluar, no cuesta tiempo)."""
    t.update(tiempo_decision_s=round(t_decision, 4), tiempo_evaluacion_s=getattr(evaluacion, "tiempo_s", None),
             repetida=repetida)
    return t


def _cambiar(malla, f, c, letra):
    return tuple(tuple(letra if (i, j) == (f, c) else t for j, t in enumerate(fila)) for i, fila in enumerate(malla))


def fase_tipos(agente, d, estado, malla, c_actual, azar, trayectoria, decisiones):
    """La fase de tipos sobre ``malla`` (ver el docstring del módulo). Devuelve ``(malla, C, parada)``."""
    p = agente.parametros
    tolerancia = agente.costos.tolerancia_empate
    if not d.alternos:
        return malla, c_actual, "sin_tipos"
    m = agente.evaluacion.metricas or {}
    c_a_ref = m["C_A"] / m["C_A_norm"] if m.get("C_A") and m.get("C_A_norm") else None
    k_a = agente.costos.K.get("K_A", 1.0)
    area = {t: agente.costos.area_pe(agente.base["tipos_pe"][t]["instrucciones"]) for t in d.alternos}
    vistas, rechazados, simuladas = {malla}, set(), 0
    while True:
        if simuladas >= p.max_cambios_tipo:
            return malla, c_actual, "tope"
        t_decision = time.monotonic()
        opciones = {}
        for f, fila in enumerate(malla):
            for c in range(1, len(fila) - 1):  # el interior: las columnas de memoria no se tocan
                de = fila[c]
                if de not in d.alternos or (malla, (f, c)) in rechazados:
                    continue
                a = d.alternos[de]
                nueva = _cambiar(malla, f, c, a)
                if nueva in vistas or not agente.cumple_tipos({t for r in nueva for t in r}):
                    continue
                estimado = c_actual + (k_a * (area[a] - area[de]) / c_a_ref if c_a_ref else 0.0)
                opciones[f"CAMBIAR {f} {c}"] = (nueva, (f, c), de, a, estimado)
        if not opciones:
            return malla, c_actual, "sin_mejora"
        orden = list(opciones)
        accion = min(orden, key=lambda x: (opciones[x][4], orden.index(x)))
        quien = "explotacion"
        if azar.random() < p.epsilon:
            accion, quien = azar.choice(orden), "exploracion"
        decisiones.append(quien)
        nueva, pe, de, a, estimado = opciones[accion]
        vistas.add(nueva)
        conf = agente.configuracion(nueva)
        es_nueva = agente._clave(conf) not in agente.cache_episodios
        t_decision = time.monotonic() - t_decision
        ev, repetida = agente.simular_episodio(conf)
        simuladas += 1
        aceptada = ev.estado == CORRECTA and ev.costo is not None and ev.costo < c_actual - tolerancia
        trayectoria.append(_tiempos(paso_tipo(d, estado, nueva, ev, accion, pe, de, a, quien, aceptada, es_nueva,
                                              estimado), t_decision, ev, repetida))
        if aceptada:
            malla, c_actual = nueva, ev.costo
        else:
            rechazados.add((malla, pe))


def descender(agente, azar):
    """Un episodio del modo ``bloques`` (ver el docstring del módulo). Devuelve el episodio."""
    d = Descenso(agente)
    p = agente.parametros
    tolerancia = agente.costos.tolerancia_empate
    estado = d.inicial()
    conf = agente.configuracion(d.malla(estado))
    clave = agente._clave(conf)
    if clave not in agente.cache_episodios and d.malla(estado) == tuple(tuple(f) for f in d.referencia):
        agente.cache_episodios[clave] = agente.evaluacion  # la referencia ya está simulada
    ev, _ = agente.simular_episodio(conf)
    c_actual = ev.costo
    trayectoria = [paso(d, estado, ev, "inicio", None, "inicio", True, False, d.estimado(estado))]
    decisiones, rechazadas = [], set()
    niveles = [b for b in p.bloques if b >= 1]
    i, parada = 0, None
    while i < len(niveles):
        t_decision = time.monotonic()
        b = niveles[i]
        opciones, mallas = {}, set()
        for accion in PREFERENCIA:
            nuevo = d.aplicar(estado, accion, b)
            if nuevo is None or not d.valido(nuevo) or (estado, accion, b) in rechazadas:
                continue
            malla_nueva = d.malla(nuevo)
            if malla_nueva in mallas:
                continue  # otra acción ya da esta misma malla (p. ej. columnas del inicio o del final)
            mallas.add(malla_nueva)
            opciones[accion] = nuevo
        if not opciones:
            i += 1
            continue
        estimados = {a: d.estimado(e) for a, e in opciones.items()}
        accion = min(opciones, key=lambda a: (estimados[a], ACCIONES.index(a)))
        quien = "explotacion"
        if azar.random() < p.epsilon:
            accion, quien = azar.choice(sorted(opciones, key=ACCIONES.index)), "exploracion"
        decisiones.append(quien)
        nuevo = opciones[accion]
        conf = agente.configuracion(d.malla(nuevo))
        nueva = agente._clave(conf) not in agente.cache_episodios
        t_decision = time.monotonic() - t_decision
        ev, repetida = agente.simular_episodio(conf)
        aceptada = ev.estado == CORRECTA and ev.costo is not None and ev.costo < c_actual - tolerancia
        trayectoria.append(_tiempos(paso(d, nuevo, ev, accion, b, quien, aceptada, nueva, estimados[accion]),
                                    t_decision, ev, repetida))
        if aceptada:
            estado, c_actual = nuevo, ev.costo
        else:
            rechazadas.add((estado, accion, b))
            for otra in ACCIONES:  # las acciones que dan la misma malla también quedan descartadas
                otro = d.aplicar(estado, otra, b)
                if otro is not None and d.malla(otro) == d.malla(nuevo):
                    rechazadas.add((estado, otra, b))
    else:
        parada = "sin_mejora"

    malla = d.malla(estado)
    parada_tipos = None
    if p.tipos:
        malla, c_actual, parada_tipos = fase_tipos(agente, d, estado, malla, c_actual, azar, trayectoria, decisiones)
    final = agente.cache_episodios.get(agente._clave(agente.configuracion(malla)))
    m = final.metricas or {}
    mejor_previo = agente.c_best if agente.c_best is not None else agente.evaluacion.costo
    finales_previos = {tuple(e["malla"]) for e in agente.historial}
    texto = [" ".join(f) for f in malla]
    simuladas = [t for t in trayectoria if t.get("estado") is not None and t["accion"] != "inicio"]
    return {
        "n": agente.n_episodio, "modo": "bloques", "semilla": agente.semilla_actual,
        "camino": [], "decisiones": decisiones, "parada": parada, "extensiones": [], "extension": 0,
        "repetido": tuple(texto) in finales_previos, "malla": texto, "filas": len(malla),
        "columnas": len(malla[0]), "pes": len(malla) * len(malla[0]), "estado": final.estado,
        "motivo": final.motivo, "C": final.costo, "C_A": m.get("C_A"), "C_A_norm": m.get("C_A_norm"),
        "C_L_norm": m.get("C_L_norm"), "C_R": m.get("C_R"),
        "costo_por_operacion": {op: o.costo for op, o in final.operaciones.items()},
        "C_estimado": d.estimado(estado), "C_best_prev": mejor_previo,
        "recompensa": recompensa(final.estado, final.costo, mejor_previo, agente.costos.p_fallo),
        "simuladas": len(simuladas), "nuevas": sum(1 for t in simuladas if t["nueva"]),
        "aceptadas": sum(1 for t in simuladas if t["aceptada"]),
        "parada_tipos": parada_tipos,
        "cambios_tipo": sum(1 for t in simuladas if t.get("fase") == "tipos" and t["aceptada"]),
        "trayectoria": trayectoria,
        "formula": "Descenso por bloques desde la malla de referencia (y, con tipos, cambios E <-> V de PEs del "
                   "interior): cada malla propuesta se simula; se acepta si su C real baja respecto a la actual. "
                   "R = C_best_prev − C(malla final).",
    }
