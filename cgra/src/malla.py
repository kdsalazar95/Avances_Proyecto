"""El hardware de la malla: los PEs y las colas que los conectan.

Cada PE es un procesador pequeño: tiene sus registros y ejecuta una
instrucción a la vez. Se comunica con sus 4 vecinos (norte, sur, este,
oeste) por colas FIFO, una por cada sentido de cada conexión, para que
los datos que van y los que vienen nunca se mezclen.
"""

import operator

from isa import ISA
from topologia import DIRECCIONES, DESPLAZAMIENTO, OPUESTA

ARITMETICA = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}
if ARITMETICA.keys() != ISA.ARITMETICAS:
    raise RuntimeError("Las operaciones aritméticas de la ISA y la malla no coinciden")


class Cola:
    """Canal FIFO de un solo sentido entre dos PEs vecinos."""

    def __init__(self):
        self.datos = []

    def enviar(self, dato):
        """Agrega un dato al final de la cola (lo usa SEND)."""
        self.datos.append(dato)

    def recibir(self):
        """Saca el dato más antiguo de la cola (lo usa RECV)."""
        if not self.datos:
            raise RuntimeError("RECV intentó leer una cola FIFO vacía")
        return self.datos.pop(0)


class PE:
    """Elemento de procesamiento: registros, una cola de salida y una de
    entrada por cada vecino, y la ejecución de una instrucción."""

    def __init__(self, pe_id):
        self.id = pe_id
        self.tipo = None
        self.instrucciones_permitidas = None
        self.registros = [0] * ISA.CANTIDAD_REGISTROS
        # Una cola por dirección; None si no hay vecino de ese lado.
        self.salidas = dict.fromkeys(DIRECCIONES)
        self.entradas = dict.fromkeys(DIRECCIONES)

    def leer_registro(self, registro):
        return self.registros[ISA.indice_registro(registro)]

    def escribir_registro(self, registro, valor):
        self.registros[ISA.indice_registro(registro)] = valor

    def ejecutar(self, instruccion, memoria=None):
        """Ejecuta una instrucción ya parseada (ver ``ISA.parsear``).

        ``memoria`` es la memoria plana de la CGRA (una lista de palabras).
        Los LD/ST usan la dirección ``addr`` que les asignó el cargador.
        """
        op = instruccion["op"]
        if self.instrucciones_permitidas is not None and op not in self.instrucciones_permitidas:
            raise RuntimeError(f"PE {self.id} es de tipo {self.tipo} y no puede ejecutar {op.upper()}")
        if op == "nop":
            return
        if op == "mov":
            self.escribir_registro(instruccion["dst"], instruccion["imm"])
        elif op in ARITMETICA:
            a = self.leer_registro(instruccion["src1"])
            b = self.leer_registro(instruccion["src2"])
            self.escribir_registro(instruccion["dst"], ARITMETICA[op](a, b))
        elif op == "ld":
            self.escribir_registro(instruccion["dst"], memoria[self._direccion(instruccion, memoria)])
        elif op == "st":
            memoria[self._direccion(instruccion, memoria)] = self.leer_registro(instruccion["src"])
        elif op == "send":
            self._cola(self.salidas, instruccion["dir"]).enviar(self.leer_registro(instruccion["src"]))
        elif op == "recv":
            self.escribir_registro(instruccion["dst"], self._cola(self.entradas, instruccion["dir"]).recibir())
        else:
            raise ValueError(f"Operación desconocida: {op}")

    def _direccion(self, instruccion, memoria):
        if memoria is None:
            raise RuntimeError(f"{instruccion['op'].upper()} requiere la memoria de la CGRA")
        if "addr" not in instruccion:
            raise RuntimeError(f"{self.id}: {instruccion['bank']}[{instruccion['index']}] no fue enlazado por el cargador")
        direccion = instruccion["addr"]
        if not 0 <= direccion < len(memoria):
            raise IndexError(f"Dirección fuera de la memoria: {direccion}")
        return direccion

    def _cola(self, colas, direccion):
        if direccion not in colas:
            raise ValueError(f"Dirección inválida: {direccion}")
        if colas[direccion] is None:
            raise RuntimeError(f"{self.id} no tiene vecino hacia {direccion}")
        return colas[direccion]

    def __repr__(self):
        return f"PE(id={self.id}, registros={self.registros})"


class Malla:
    """Malla filas x columnas de PEs, cada uno conectado con sus vecinos.

    Los PEs se nombran ``PE{fila}{columna}``, igual que sus archivos
    (``PE10.txt``). Con 11 o más filas o columnas cada coordenada lleva
    dos dígitos (``PE0100`` = fila 1, columna 0).
    """

    def __init__(self, filas, columnas):
        self.filas = filas
        self.columnas = columnas
        self._ancho = len(str(max(filas, columnas) - 1))
        self._pes = [[PE(self.id_de(f, c)) for c in range(columnas)] for f in range(filas)]
        self._conectar()

    def id_de(self, fila, columna):
        """``(1, 0)`` -> ``PE10``."""
        return f"PE{fila:0{self._ancho}d}{columna:0{self._ancho}d}"

    @staticmethod
    def posicion(pe_id):
        """``PE10`` -> ``(1, 0)``. Las dos coordenadas tienen el mismo ancho."""
        numeros = pe_id[2:]
        ancho = len(numeros) // 2
        return int(numeros[:ancho]), int(numeros[ancho:])

    def ids(self):
        """IDs de todos los PEs, fila por fila: ``("PE00", "PE01", ...)``."""
        return tuple(self.id_de(f, c) for f in range(self.filas) for c in range(self.columnas))

    def pe(self, pe_id):
        fila, columna = self.posicion(pe_id)
        return self._pes[fila][columna]

    def vecino(self, pe_id, direccion):
        """ID del vecino de un PE en esa dirección."""
        if direccion not in DESPLAZAMIENTO:
            raise ValueError(f"Dirección inválida: {direccion}")
        fila, columna = self.posicion(pe_id)
        delta_fila, delta_columna = DESPLAZAMIENTO[direccion]
        destino_fila, destino_columna = fila + delta_fila, columna + delta_columna
        if not (0 <= destino_fila < self.filas and 0 <= destino_columna < self.columnas):
            raise ValueError(f"{pe_id}: no tiene vecino hacia {direccion}")
        return self.id_de(destino_fila, destino_columna)

    def _conectar(self):
        """Une cada PE con su vecino del este y del sur, con una cola por sentido."""
        for f in range(self.filas):
            for c in range(self.columnas):
                if c + 1 < self.columnas:
                    self._unir(self._pes[f][c], self._pes[f][c + 1], "east")
                if f + 1 < self.filas:
                    self._unir(self._pes[f][c], self._pes[f + 1][c], "south")

    @staticmethod
    def _unir(origen, destino, direccion):
        """Crea las dos colas entre ``origen`` y su vecino ``destino``."""
        ida, vuelta = Cola(), Cola()
        origen.salidas[direccion] = destino.entradas[OPUESTA[direccion]] = ida
        destino.salidas[OPUESTA[direccion]] = origen.entradas[direccion] = vuelta
