"""Formato de la CGRA: tamaño de la malla, tamaño de la memoria de datos y
tipo de cada PE, leídos de un JSON (por defecto ``src/cgra.json``).

El mismo archivo lo leen los skills, así que Python y el compilador
siempre trabajan con la misma CGRA.

El tipo de cada PE se da con una regla, ``distribucion``, para que al
cambiar ``filas`` y ``columnas`` no haya que reescribir la malla:

    "distribucion": {"oeste": "M", "este": "M", "resto": "C"}

pone PEs de memoria (M) en las columnas oeste y este y de cómputo (C) en
el resto, sea cual sea el tamaño. Los bordes posibles son ``norte``,
``sur``, ``oeste`` y ``este``; en una esquina mandan ``oeste``/``este``.

Para una distribución especial se puede dar la malla completa en
``malla`` (una fila por línea, ``"M C C M"``), que reemplaza a la regla.
Cada letra es un tipo definido en ``tipos_pe``, con las instrucciones que
ese tipo de PE puede ejecutar.
"""

import json
from pathlib import Path

from programas_pe import descomponer_pe_id, generar_pe_ids

RUTA_POR_DEFECTO = Path(__file__).with_name("cgra.json")

BORDES = ("norte", "sur", "oeste", "este")

# Instrucciones que existen en el ISA (las que entiende parsear_instruccion).
INSTRUCCIONES_ISA = ("NOP", "MOV", "LD", "ST", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV")


class FormatoCGRA:
    """Descripción de una CGRA: malla, memoria y tipos de PE."""

    def __init__(self, filas, columnas, palabras_memoria, tipos_pe, malla):
        """``tipos_pe``: ``{letra: {"nombre", "descripcion", "instrucciones"}}``.
        ``malla``: lista de filas, cada una con la letra del tipo de cada PE."""
        self.filas = filas
        self.columnas = columnas
        self.palabras_memoria = palabras_memoria
        self.tipos_pe = tipos_pe
        self.malla = malla
        self._validar()

    def _validar(self):
        if self.filas <= 0 or self.columnas <= 0:
            raise ValueError("La malla debe tener al menos una fila y una columna")
        if self.palabras_memoria <= 0:
            raise ValueError("La memoria de la CGRA debe tener al menos una palabra")
        if not self.tipos_pe:
            raise ValueError("Falta definir al menos un tipo de PE en 'tipos_pe'")
        for letra, tipo in self.tipos_pe.items():
            desconocidas = set(tipo["instrucciones"]) - set(INSTRUCCIONES_ISA)
            if desconocidas:
                raise ValueError(
                    f"El tipo '{letra}' usa instrucciones que no existen: {', '.join(sorted(desconocidas))}"
                )
        if len(self.malla) != self.filas:
            raise ValueError(f"'malla' tiene {len(self.malla)} filas y 'filas' es {self.filas}")
        for f, fila in enumerate(self.malla):
            if len(fila) != self.columnas:
                raise ValueError(f"La fila {f} de 'malla' tiene {len(fila)} PEs y 'columnas' es {self.columnas}")
            for c, letra in enumerate(fila):
                if letra not in self.tipos_pe:
                    raise ValueError(f"PE({f},{c}): tipo '{letra}' no está definido en 'tipos_pe'")

    def letra(self, pe_id):
        """Letra del tipo de un PE (por ejemplo ``"M"``)."""
        fila, columna = descomponer_pe_id(pe_id)
        return self.malla[fila][columna]

    def nombre_tipo(self, pe_id):
        """Nombre del tipo de un PE (por ejemplo ``"memoria"``)."""
        return self.tipos_pe[self.letra(pe_id)]["nombre"]

    def instrucciones(self, pe_id):
        """Instrucciones que puede ejecutar un PE, en minúsculas como ``inst["op"]``."""
        return {op.lower() for op in self.tipos_pe[self.letra(pe_id)]["instrucciones"]}

    def pe_ids(self):
        return generar_pe_ids(self.filas, self.columnas)

    def mapa(self):
        """Líneas de texto con la malla, una por fila (``M C C M``)."""
        return [" ".join(fila) for fila in self.malla]

    def leyenda(self):
        """Una línea por tipo: letra, nombre e instrucciones."""
        return [
            f"{letra} = {tipo['nombre']}: {', '.join(tipo['instrucciones'])}"
            for letra, tipo in self.tipos_pe.items()
        ]


def generar_malla(filas, columnas, distribucion):
    """Arma la malla a partir de la regla ``distribucion``.

    Cada PE toma el tipo de su borde (``norte``, ``sur``, ``oeste``,
    ``este``) o, si no está en un borde con tipo, el de ``resto``. En una
    esquina mandan ``oeste``/``este``, que son las columnas de memoria de la
    CGRA por defecto.
    """
    desconocidas = set(distribucion) - set(BORDES) - {"resto"}
    if desconocidas:
        raise ValueError(
            f"'distribucion' tiene claves desconocidas: {', '.join(sorted(desconocidas))} "
            f"(se usan {', '.join(BORDES)} y resto)"
        )
    if "resto" not in distribucion:
        raise ValueError("'distribucion' necesita 'resto': el tipo de los PEs que no están en un borde")

    def tipo(f, c):
        if c == 0 and "oeste" in distribucion:
            return distribucion["oeste"]
        if c == columnas - 1 and "este" in distribucion:
            return distribucion["este"]
        if f == 0 and "norte" in distribucion:
            return distribucion["norte"]
        if f == filas - 1 and "sur" in distribucion:
            return distribucion["sur"]
        return distribucion["resto"]

    return [[tipo(f, c) for c in range(columnas)] for f in range(filas)]


def cargar_formato(ruta=RUTA_POR_DEFECTO):
    """Lee y valida el JSON con el formato de la CGRA."""
    ruta = Path(ruta)
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"No existe el formato de la CGRA: {ruta}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{ruta}: JSON inválido ({error})") from None

    faltan = [campo for campo in ("filas", "columnas", "palabras_memoria", "tipos_pe") if campo not in datos]
    if "malla" not in datos and "distribucion" not in datos:
        faltan.append("distribucion (o malla)")
    if faltan:
        raise ValueError(f"{ruta}: faltan los campos {', '.join(faltan)}")

    tipos_pe = {}
    for letra, tipo in datos["tipos_pe"].items():
        if len(letra) != 1 or not letra.isalpha():
            raise ValueError(f"{ruta}: el tipo '{letra}' debe ser una sola letra")
        tipos_pe[letra] = {
            "nombre": tipo.get("nombre", letra),
            "descripcion": tipo.get("descripcion", ""),
            "instrucciones": [op.upper() for op in tipo["instrucciones"]],
        }
    try:
        if "malla" in datos:
            # Malla explícita: cada fila como texto ("M C C M"); los espacios son solo visuales.
            malla = [fila.split() if isinstance(fila, str) else list(fila) for fila in datos["malla"]]
        else:
            malla = generar_malla(datos["filas"], datos["columnas"], datos["distribucion"])
        return FormatoCGRA(datos["filas"], datos["columnas"], datos["palabras_memoria"], tipos_pe, malla)
    except ValueError as error:
        raise ValueError(f"{ruta}: {error}") from None


def validar_tipos(programas, formato):
    """Verifica que cada PE solo use instrucciones de su tipo.

    Junta todos los errores (hasta 10 en el mensaje) para que se vea de
    una vez qué PEs y ciclos hay que corregir en el schedule.
    """
    errores = []
    for pe_id, programa in programas.items():
        permitidas = formato.instrucciones(pe_id)
        for ciclo in sorted(programa):
            op = programa[ciclo]["op"]
            if op not in permitidas:
                errores.append(
                    f"{pe_id} ({formato.nombre_tipo(pe_id)}), ciclo {ciclo}: {op.upper()} no está permitido"
                )
    if errores:
        detalle = "\n  ".join(errores[:10])
        extra = f"\n  ... y {len(errores) - 10} más" if len(errores) > 10 else ""
        raise ValueError(
            f"{len(errores)} instrucciones no respetan el tipo de su PE:\n  {detalle}{extra}"
        )
