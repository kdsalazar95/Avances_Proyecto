"""Malla y compatibilidad del bloque Dijkstra (skills 05 y 06).

``MallaMapeo`` es el grafo de la malla que usa el algoritmo: cada PE con su
fila, columna, letra de tipo y lista de instrucciones, y sus vecinos en
orden fijo (norte, sur, oeste, este). Se arma desde el ``FormatoCGRA`` de
``cgra/src/cgra.json`` (tipos, ``distribucion``/``ciclo_filas`` o ``malla``),
así que la malla es exactamente la que carga el simulador.

``Compatibilidad`` traduce cada operación del grafo a una instrucción del
ISA (``compatibilidad.json``); los tipos que la ejecutan son los que
tienen esa instrucción en su lista ``instrucciones``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

RUTA_COMPATIBILIDAD = Path(__file__).resolve().parents[1] / "config" / "compatibilidad.json"

# Orden fijo de los vecinos y el nombre de cada dirección en los PE*.txt.
VECINOS = (("north", -1, 0), ("south", 1, 0), ("west", 0, -1), ("east", 0, 1))
OPUESTA = {"north": "south", "south": "north", "west": "east", "east": "west"}
ARITMETICA = frozenset({"ADD", "SUB", "MUL", "DIV"})


@dataclass(frozen=True)
class PEMapeo:
    indice: int
    id: str
    fila: int
    columna: int
    letra: str
    nombre_tipo: str
    instrucciones: frozenset


class MallaMapeo:
    """La malla como grafo: PEs (en orden de filas) y sus vecinos."""

    def __init__(self, filas, columnas, letras, tipos, registros=("rA", "rB", "rC", "rT", "acc"),
                 registros_vectoriales=tuple(f"v{i}" for i in range(8))):
        """``letras``: una lista de filas con la letra de cada PE;
        ``tipos``: ``{letra: {"nombre": ..., "instrucciones": [...]}}``.
        ``registros_vectoriales`` queda vacío si la CGRA no tiene ``ancho_vector``."""
        if filas < 1 or columnas < 1:
            raise ValueError("la malla necesita al menos una fila y una columna")
        if len(letras) != filas or any(len(fila) != columnas for fila in letras):
            raise ValueError(f"el mapa de tipos no es de {filas}x{columnas}")
        self.filas = filas
        self.columnas = columnas
        self.registros = tuple(registros)
        self.registros_vectoriales = tuple(registros_vectoriales)
        ancho = len(str(max(filas, columnas) - 1))
        self.pes = []
        for f in range(filas):
            for c in range(columnas):
                letra = letras[f][c]
                if letra not in tipos:
                    raise ValueError(f"PE({f},{c}): el tipo '{letra}' no está definido")
                self.pes.append(PEMapeo(
                    indice=f * columnas + c, id=f"PE{f:0{ancho}d}{c:0{ancho}d}", fila=f, columna=c, letra=letra,
                    nombre_tipo=tipos[letra].get("nombre", letra),
                    instrucciones=frozenset(i.upper() for i in tipos[letra]["instrucciones"]),
                ))
        self.vecinos = []
        for pe in self.pes:
            lista = []
            for direccion, df, dc in VECINOS:
                f, c = pe.fila + df, pe.columna + dc
                if 0 <= f < filas and 0 <= c < columnas:
                    lista.append((direccion, f * columnas + c))
            self.vecinos.append(tuple(lista))
        self.direccion = {(i, j): d for i, lista in enumerate(self.vecinos) for d, j in lista}

    @classmethod
    def desde_formato(cls, formato):
        """Desde un ``FormatoCGRA`` (lo que lee el simulador de cgra.json)."""
        from isa import ISA
        tipos = {letra: {"nombre": t["nombre"], "instrucciones": t["instrucciones"]}
                 for letra, t in formato.tipos_pe.items()}
        vectoriales = tuple(ISA.VECTORIALES) if formato.ancho_vector else ()
        return cls(formato.filas, formato.columnas, formato.malla, tipos, tuple(ISA.REGISTROS), vectoriales)

    def __len__(self):
        return len(self.pes)

    def puede(self, indice, instruccion):
        return instruccion in self.pes[indice].instrucciones

    def distancia(self, i, j):
        a, b = self.pes[i], self.pes[j]
        return abs(a.fila - b.fila) + abs(a.columna - b.columna)

    def es_memoria(self, indice):
        """PE que accede a memoria y no calcula (el tipo M del proyecto)."""
        instrucciones = self.pes[indice].instrucciones
        return bool(instrucciones & {"LD", "ST"}) and not instrucciones & ARITMETICA

    def mapa(self):
        return [" ".join(self.pes[f * self.columnas + c].letra for c in range(self.columnas))
                for f in range(self.filas)]


class Compatibilidad:
    """Operación del grafo -> instrucción del ISA -> PEs que la pueden ejecutar."""

    def __init__(self, operaciones, transferencia):
        self.operaciones = {op: instruccion.upper() for op, instruccion in operaciones.items()}
        self.enviar = transferencia["enviar"].upper()
        self.recibir = transferencia["recibir"].upper()
        self.enviar_vector = transferencia.get("enviar_vector", "").upper()
        self.recibir_vector = transferencia.get("recibir_vector", "").upper()

    def transferencia(self, vectorial):
        """``(enviar, recibir)`` para un escalar o un vector."""
        return (self.enviar_vector, self.recibir_vector) if vectorial else (self.enviar, self.recibir)

    @classmethod
    def cargar(cls, ruta=RUTA_COMPATIBILIDAD):
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(datos["operaciones"], datos["transferencia"])

    def instruccion(self, op):
        return self.operaciones.get(op)

    def candidatos(self, malla, op):
        """Índices de los PEs que ejecutan ``op``, en orden; ``None`` si ``op`` no está en la tabla."""
        instruccion = self.instruccion(op)
        if instruccion is None:
            return None
        return [pe.indice for pe in malla.pes if instruccion in pe.instrucciones]
