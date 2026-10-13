"""Traza de una ejecución: qué instrucción ejecutó cada PE en cada ciclo
y qué dejó escrito. La CGRA la llena mientras ejecuta; no depende de qué
operación se está corriendo."""

from consola import Consola
from formato_numero import formatear_numero
from isa import ISA


class Traza:
    """Registro, ciclo por ciclo, de lo que hizo cada PE (sin los NOP)."""

    def __init__(self):
        # Una entrada por instrucción ejecutada:
        # {"ciclo", "pe", "instruccion", "efecto"}.
        self.entradas = []

    def registrar(self, ciclo, pe, instruccion, memoria):
        """Anota una instrucción recién ejecutada por ``pe``."""
        if instruccion["op"] == "nop":
            return
        self.entradas.append({
            "ciclo": ciclo,
            "pe": pe.id,
            "instruccion": ISA.texto(instruccion),
            "efecto": self._efecto(pe, instruccion, memoria),
        })

    @staticmethod
    def _efecto(pe, instruccion, memoria):
        """El registro o la posición de memoria que escribió, o lo que envió."""
        op = instruccion["op"]
        if op == "st":
            direccion = instruccion["addr"]
            return (f"{instruccion['bank']}[{instruccion['index']}] = "
                    f"{formatear_numero(memoria[direccion])}  (dirección {direccion})")
        if op == "send":
            return f"envía {formatear_numero(pe.leer_registro(instruccion['src']))}"
        if "dst" in instruccion:
            return f"{instruccion['dst']} = {formatear_numero(pe.leer_registro(instruccion['dst']))}"
        return ""

    def lineas(self):
        """``ciclo | PE | instrucción | efecto``, una línea por entrada."""
        return [
            f"  {e['ciclo']:>5} | {e['pe']} | {e['instruccion']:<22} | {e['efecto']}"
            for e in self.entradas
        ]

    def mostrar(self):
        Consola.titulo("TRAZA: ciclo | PE | instrucción | efecto")
        for linea in self.lineas():
            print(linea)
