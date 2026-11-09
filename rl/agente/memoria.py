"""A05 · Historial de evaluaciones (historial.jsonl) y entrada de la política.

El historial es de solo agregar; las óptimas, el mejor candidato y la paciencia
se reconstruyen leyéndolo (reanudación y auditoría). El LLM nunca escribe en él
ni lo recibe: la política del enunciado es π(C, E) = A y solo ve los costos de
su última acción y el mejor candidato actual (``entrada_politica``).
"""

from collections import Counter
from pathlib import Path

from .. import estados
from ..comun import agregar_jsonl, ahora, leer_jsonl
from .optimas import actualizar_optimas


class Memoria:
    def __init__(self, ruta_historial, tolerancia_empate=0.0):
        self.ruta = Path(ruta_historial)
        self.tolerancia = tolerancia_empate
        self.entradas = leer_jsonl(self.ruta)

    # --- actualizar -------------------------------------------------------
    def agregar(self, reporte, iteracion, recibido=None, malla=None):
        """Agrega una evaluación completa (reporte E08 + iteración + marca de tiempo)."""
        if any(e["id_evaluacion"] == reporte["id_evaluacion"] for e in self.entradas):
            raise ValueError(f"'{reporte['id_evaluacion']}' ya está en el historial")
        entrada = dict(reporte)
        entrada["iteracion"] = iteracion
        entrada["marca_tiempo"] = ahora()
        entrada["duplicada_de"] = self.ya_evaluada(reporte["accion"].get("hash_accion"))
        entrada["malla"] = malla
        if recibido is not None:
            entrada["recibido_A04"] = {k: recibido[k] for k in ("C", "r", "e")}
        agregar_jsonl(self.ruta, entrada)
        self.entradas.append(entrada)
        return entrada

    # --- consultar --------------------------------------------------------
    def por_id(self, id_evaluacion):
        return next((e for e in self.entradas if e["id_evaluacion"] == id_evaluacion), None)

    def ya_evaluada(self, hash_accion):
        """id de la primera evaluación con el mismo paquete ejecutable, o None."""
        if not hash_accion:
            return None
        return next((e["id_evaluacion"] for e in self.entradas
                     if e["accion"].get("hash_accion") == hash_accion), None)

    def conteo(self):
        return dict(Counter(e["estado_evaluacion"] for e in self.entradas))

    def mejor_costo(self):
        correctos = [e["costo_total"] for e in self.entradas if e["estado_evaluacion"] == estados.CORRECTA]
        return min(correctos, default=None)

    def siguiente_iteracion(self):
        return max((e["iteracion"] for e in self.entradas), default=-1) + 1

    # --- reconstrucción ---------------------------------------------------
    def reconstruir(self):
        """Óptimas, C_min y contador sin mejora a partir del historial (reanudar)."""
        optimas, c_min, sin_mejora = [], None, 0
        for entrada in self.entradas:
            optimas, c_min, mejora = actualizar_optimas(optimas, entrada, self.tolerancia)
            # La referencia y los fallos de infraestructura no cuentan para la paciencia.
            if not entrada.get("es_referencia") and entrada["estado_evaluacion"] not in estados.NO_ATRIBUIBLES:
                sin_mejora = 0 if mejora else sin_mejora + 1
        return {"optimas": optimas, "C_min": c_min, "sin_mejora": sin_mejora}

    # --- entrada de la política π(C, E) = A ------------------------------
    @staticmethod
    def _costos(e):
        m = e["metricas"]
        return {"id": e["id_evaluacion"], "malla": e.get("malla"), "estado": e["estado_evaluacion"],
                "C_A": m.get("C_A"), "C_L": m.get("C_L"), "C_R": m.get("C_R"), "C": e["costo_total"]}

    @staticmethod
    def _mejor(entradas):
        correctas = [e for e in entradas if e["estado_evaluacion"] == estados.CORRECTA]
        return min(correctas, key=lambda e: (e["costo_total"], e["iteracion"]), default=None)

    def mejor_candidato(self):
        """El mejor candidato actual: el primero que alcanzó el menor costo entre las correctas.

        Regla del enunciado: si una acción da un costo más bajo, pasa a ser el mejor candidato; si da un
        costo mayor (o igual), se descarta.
        """
        return self._mejor(self.entradas)

    def _punto_de_partida(self, entrada):
        """El mejor candidato contra el que se midió ``entrada`` (el de C_best_prev de su recompensa).

        En un lote todas se miden contra el mejor anterior al lote, así que no basta con mirar la entrada
        inmediatamente anterior.
        """
        anteriores = [e for e in self.entradas if e["iteracion"] < entrada["iteracion"]]
        c_prev = (entrada.get("detalle_recompensa") or {}).get("C_best_prev")
        if c_prev is not None:
            iguales = [e for e in anteriores if e["estado_evaluacion"] == estados.CORRECTA
                       and abs(e["costo_total"] - c_prev) <= max(self.tolerancia, 1e-12)]
            if iguales:
                return min(iguales, key=lambda e: e["iteracion"])
        return self._mejor(anteriores)

    @staticmethod
    def _cambio(antes, despues):
        cambio = despues - antes if isinstance(antes, (int, float)) and isinstance(despues, (int, float)) else None
        return {"antes": antes, "despues": despues, "cambio": cambio}

    def entrada_politica(self):
        """Lo que recibe la política π(C, E) = A del enunciado (y nada más del historial).

        * ``E``: el estado, es decir, los cambios que produjo la última acción respecto al mejor candidato
          del que partió: la malla antes y después, y C_A, C_L, C_R y C antes, después y su diferencia;
          más la recompensa r con la que se la premió o penalizó y su estado de evaluación.
        * ``mejor_candidato``: la configuración con menor costo hasta ahora y sus costos.
        """
        if not self.entradas:
            return None
        ultima = self.entradas[-1]
        despues = self._costos(ultima)
        partida = self._punto_de_partida(ultima)
        antes = self._costos(partida) if partida else {}
        estado = {
            "id": despues["id"],
            "estado": despues["estado"],
            "partio_de": antes.get("id"),
            "malla": {"antes": antes.get("malla"), "despues": despues["malla"]},
            **{k: self._cambio(antes.get(k), despues[k]) for k in ("C_A", "C_L", "C_R", "C")},
            "r": ultima["recompensa"],
        }
        mejor = self.mejor_candidato()
        return {"E": estado, "mejor_candidato": self._costos(mejor) if mejor else None}
