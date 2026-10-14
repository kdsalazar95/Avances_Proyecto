"""El programa de la CGRA: un PE{fila}{columna}.txt por PE (skill 08).

Cada archivo tiene una instrucción por ciclo:

    00: MOV acc, 0.0
    01: LD rA, a[0]     // los comentarios empiezan con //

Todos los PEs deben cubrir los mismos ciclos, y cada SEND debe tener en
el mismo ciclo el RECV de su vecino (y al revés); lo mismo VSEND y VRECV.
"""

import re
from pathlib import Path

from isa import ISA
from malla import OPUESTA

PATRON_LINEA = re.compile(r"^(\d+):\s*(.*)$")  # "05: ADD rC, rA, rB"


class Programa:
    """Las instrucciones de todos los PEs: ``programa[pe_id][ciclo]``."""

    def __init__(self, instrucciones):
        """``instrucciones``: ``{pe_id: {ciclo: instrucción parseada}}``."""
        self.instrucciones = instrucciones

    @classmethod
    def leer(cls, directorio, malla):
        """Lee y valida los PE*.txt de ``directorio`` para esa malla."""
        directorio = Path(directorio)
        instrucciones = {}
        for pe_id in malla.ids():
            ruta = directorio / f"{pe_id}.txt"
            if not ruta.is_file():
                raise FileNotFoundError(f"No existe {ruta}")
            instrucciones[pe_id] = cls._leer_archivo(ruta)
        programa = cls(instrucciones)
        programa.validar(malla)
        return programa

    @staticmethod
    def _leer_archivo(ruta):
        """Lee un PE*.txt y devuelve ``{ciclo: instrucción}``."""
        instrucciones = {}
        for numero_linea, linea in enumerate(Path(ruta).read_text(encoding="utf-8").splitlines(), start=1):
            texto = linea.split("//", 1)[0].strip()
            if not texto:
                continue
            coincidencia = PATRON_LINEA.fullmatch(texto)
            if not coincidencia:
                raise ValueError(f"{ruta}, línea {numero_linea}: se esperaba 'NN: INSTRUCCIÓN'")
            ciclo = int(coincidencia.group(1))
            if ciclo in instrucciones:
                raise ValueError(f"{ruta}, ciclo {ciclo}: ciclo duplicado")
            instrucciones[ciclo] = ISA.parsear(coincidencia.group(2), f"{ruta}, ciclo {ciclo}")
        if not instrucciones:
            raise ValueError(f"{ruta}: programa vacío")
        return instrucciones

    def __getitem__(self, pe_id):
        return self.instrucciones[pe_id]

    def ciclos(self):
        """Los ciclos del programa, en orden (0, 1, 2, ...)."""
        return sorted(next(iter(self.instrucciones.values())))

    def del_ciclo(self, ciclo):
        """``{pe_id: instrucción}`` de un ciclo."""
        return {pe_id: instrucciones[ciclo] for pe_id, instrucciones in self.instrucciones.items()}

    def validar(self, malla):
        """Ciclos consecutivos e iguales en todos los PEs, y SEND/RECV (VSEND/VRECV) emparejados."""
        ciclos = set(self.ciclos())
        if ciclos != set(range(max(ciclos) + 1)):
            raise ValueError("Los ciclos deben comenzar en 0 y ser consecutivos")
        for pe_id, instrucciones in self.instrucciones.items():
            if set(instrucciones) != ciclos:
                raise ValueError(f"{pe_id}: rango de ciclos distinto al resto")

        for ciclo in sorted(ciclos):
            del_ciclo = self.del_ciclo(ciclo)
            for pe_id, instruccion in del_ciclo.items():
                if instruccion["op"] not in ISA.COMUNICACION:
                    continue
                otro = malla.vecino(pe_id, instruccion["dir"])
                esperada = ISA.PAREJA[instruccion["op"]]
                pareja = del_ciclo[otro]
                if pareja["op"] != esperada or pareja["dir"] != OPUESTA[instruccion["dir"]]:
                    raise ValueError(
                        f"Ciclo {ciclo}: {pe_id} {instruccion['op'].upper()} {instruccion['dir']} no coincide "
                        f"con {otro} {esperada.upper()} {OPUESTA[instruccion['dir']]}"
                    )

    def enlazar(self, simbolos, ancho_vector=None):
        """Le pone a cada LD/ST (VLD/VST) la dirección de su palabra en la memoria plana.

        ``a[3]`` pasa a ser la dirección de ``a`` + 3, según la tabla de
        símbolos de lo que el cargador copió a la memoria. Un VLD/VST usa
        ``ancho_vector`` palabras desde ahí, y todas deben caber en la región.
        """
        for pe_id, instrucciones in self.instrucciones.items():
            for ciclo, instruccion in instrucciones.items():
                if instruccion["op"] not in ISA.MEMORIA:
                    continue
                palabras = ancho_vector if instruccion["op"] in {"vld", "vst"} else 1
                if not palabras:
                    raise ValueError(f"{pe_id}, ciclo {ciclo}: {instruccion['op'].upper()} necesita 'ancho_vector' en cgra.json")
                banco, indice = instruccion["bank"], instruccion["index"]
                if banco not in simbolos:
                    raise ValueError(f"{pe_id}, ciclo {ciclo}: la región '{banco}' no está en la memoria cargada")
                direccion, filas, columnas = simbolos[banco]
                if not (0 <= indice and indice + palabras <= filas * columnas):
                    ocupa = f"{banco}[{indice}]" if palabras == 1 else f"{banco}[{indice}..{indice + palabras - 1}]"
                    raise ValueError(
                        f"{pe_id}, ciclo {ciclo}: {ocupa} se sale de la región ({filas * columnas} palabras)"
                    )
                instruccion["addr"] = direccion + indice
