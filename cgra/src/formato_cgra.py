"""Formato de la CGRA: tamaño de la malla y tamaño de la memoria de datos,
leídos de un JSON (por defecto ``src/cgra.json``).

El mismo archivo lo leen los skills, así que Python y el compilador
siempre trabajan con la misma CGRA.

La CGRA es homogénea: todos los PEs son iguales y ejecutan todo el ISA.
Lo que cambie de un PE a otro (por ejemplo, su tipo) se agrega aquí, en
el formato, para que la CGRA y los skills lo lean del mismo lugar.
"""

import json
from pathlib import Path

RUTA_POR_DEFECTO = Path(__file__).with_name("cgra.json")


class FormatoCGRA:
    """Descripción de una CGRA: tamaño de la malla y de la memoria."""

    CAMPOS = ("filas", "columnas", "palabras_memoria")

    def __init__(self, filas, columnas, palabras_memoria):
        if filas <= 0 or columnas <= 0:
            raise ValueError("La malla debe tener al menos una fila y una columna")
        if palabras_memoria <= 0:
            raise ValueError("La memoria de la CGRA debe tener al menos una palabra")
        self.filas = filas
        self.columnas = columnas
        self.palabras_memoria = palabras_memoria

    @classmethod
    def cargar(cls, ruta=RUTA_POR_DEFECTO):
        """Lee y valida el JSON con el formato de la CGRA."""
        ruta = Path(ruta)
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"No existe el formato de la CGRA: {ruta}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"{ruta}: JSON inválido ({error})") from None

        faltan = [campo for campo in cls.CAMPOS if campo not in datos]
        if faltan:
            raise ValueError(f"{ruta}: faltan los campos {', '.join(faltan)}")
        no_enteros = [campo for campo in cls.CAMPOS if type(datos[campo]) is not int]
        if no_enteros:
            raise ValueError(f"{ruta}: {', '.join(no_enteros)} deben ser números enteros")
        try:
            return cls(datos["filas"], datos["columnas"], datos["palabras_memoria"])
        except ValueError as error:
            raise ValueError(f"{ruta}: {error}") from None
