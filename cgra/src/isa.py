"""El conjunto de instrucciones (ISA) de la CGRA.

Define qué instrucciones y qué registros tiene cada PE, y traduce entre
la sintaxis de los PE*.txt (``ADD rC, rA, rB``) y el diccionario que
ejecuta ``PE.ejecutar()`` (``{"op": "add", "dst": "rC", ...}``).

Las instrucciones vectoriales (``V...``) operan sobre registros ``v0..v7``
de ``ancho_vector`` palabras (cgra.json). ``VLD v0, a[4]`` lee las
``ancho_vector`` palabras seguidas que empiezan en ``a[4]``, y ``VSEND`` /
``VRECV`` mueven el vector completo en un ciclo.
"""

import re

from formato_numero import formatear_numero

PATRON_MEMORIA = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")  # a[3] -> ("a", 3)


class ISA:
    """Instrucciones y registros de un PE."""

    # Cada PE tiene 16 registros; estos son los que tienen nombre.
    CANTIDAD_REGISTROS = 16
    REGISTROS = {"rA": 0, "rB": 1, "rC": 2, "rT": 3, "acc": 4}

    # Registros vectoriales: cada uno guarda ancho_vector palabras.
    CANTIDAD_VECTORIALES = 8
    VECTORIALES = {f"v{i}": i for i in range(CANTIDAD_VECTORIALES)}

    # En el orden en que se muestran (leyenda de cgra.json).
    INSTRUCCIONES_ESCALARES = ("NOP", "MOV", "LD", "ST", "ADD", "SUB", "MUL", "DIV", "SEND", "RECV")
    INSTRUCCIONES_VECTORIALES = (
        "VLD", "VST", "VADD", "VSUB", "VMUL", "VDIV", "VREDSUM", "VBCAST", "VSEND", "VRECV",
    )
    INSTRUCCIONES = INSTRUCCIONES_ESCALARES + INSTRUCCIONES_VECTORIALES

    ARITMETICAS = frozenset({"add", "sub", "mul", "div"})
    ARITMETICAS_VECTORIALES = frozenset("v" + op for op in ARITMETICAS)
    MEMORIA = frozenset({"ld", "st", "vld", "vst"})
    # Cada envío se empareja en el mismo ciclo con su recepción en el vecino.
    PAREJA = {"send": "recv", "recv": "send", "vsend": "vrecv", "vrecv": "vsend"}
    COMUNICACION = frozenset(PAREJA)
    ENVIOS = frozenset({"send", "vsend"})

    @staticmethod
    def parsear(texto, origen="instrucción"):
        """Convierte ``LD rA, a[3]`` en ``{"op": "ld", "dst": "rA", "bank": "a", "index": 3}``.

        ``origen`` (por ejemplo ``PE00.txt, ciclo 5``) solo se usa en los
        mensajes de error.
        """
        partes = texto.replace(",", " ").split()
        if not partes:
            raise ValueError(f"{origen}: instrucción vacía")
        op, argumentos = partes[0].upper(), partes[1:]

        if op == "NOP" and not argumentos:
            return {"op": "nop"}
        if op == "MOV" and len(argumentos) == 2:
            return {"op": "mov", "dst": argumentos[0], "imm": ISA._constante(argumentos[1])}
        if op == "LD" and len(argumentos) == 2:
            banco, indice = ISA._operando_memoria(argumentos[1], origen)
            return {"op": "ld", "dst": argumentos[0], "bank": banco, "index": indice}
        if op == "ST" and len(argumentos) == 2:
            banco, indice = ISA._operando_memoria(argumentos[0], origen)
            return {"op": "st", "bank": banco, "index": indice, "src": argumentos[1]}
        if op.lower() in ISA.ARITMETICAS and len(argumentos) == 3:
            return {"op": op.lower(), "dst": argumentos[0], "src1": argumentos[1], "src2": argumentos[2]}
        if op == "SEND" and len(argumentos) == 2:
            return {"op": "send", "dir": argumentos[0].lower(), "src": argumentos[1]}
        if op == "RECV" and len(argumentos) == 2:
            return {"op": "recv", "dir": argumentos[0].lower(), "dst": argumentos[1]}
        if op == "VLD" and len(argumentos) == 2:
            banco, indice = ISA._operando_memoria(argumentos[1], origen)
            return {"op": "vld", "dst": argumentos[0], "bank": banco, "index": indice}
        if op == "VST" and len(argumentos) == 2:
            banco, indice = ISA._operando_memoria(argumentos[0], origen)
            return {"op": "vst", "bank": banco, "index": indice, "src": argumentos[1]}
        if op.lower() in ISA.ARITMETICAS_VECTORIALES and len(argumentos) == 3:
            return {"op": op.lower(), "dst": argumentos[0], "src1": argumentos[1], "src2": argumentos[2]}
        if op in {"VREDSUM", "VBCAST"} and len(argumentos) == 2:
            return {"op": op.lower(), "dst": argumentos[0], "src": argumentos[1]}
        if op == "VSEND" and len(argumentos) == 2:
            return {"op": "vsend", "dir": argumentos[0].lower(), "src": argumentos[1]}
        if op == "VRECV" and len(argumentos) == 2:
            return {"op": "vrecv", "dir": argumentos[0].lower(), "dst": argumentos[1]}
        raise ValueError(f"{origen}: sintaxis no válida: '{texto}'")

    @staticmethod
    def texto(instruccion):
        """Lo contrario de ``parsear``: vuelve a escribir la instrucción."""
        op = instruccion["op"]
        if op == "nop":
            return "NOP"
        if op == "mov":
            return f"MOV {instruccion['dst']}, {formatear_numero(instruccion['imm'])}"
        if op in {"ld", "vld"}:
            return f"{op.upper()} {instruccion['dst']}, {instruccion['bank']}[{instruccion['index']}]"
        if op in {"st", "vst"}:
            return f"{op.upper()} {instruccion['bank']}[{instruccion['index']}], {instruccion['src']}"
        if op in ISA.ARITMETICAS or op in ISA.ARITMETICAS_VECTORIALES:
            return f"{op.upper()} {instruccion['dst']}, {instruccion['src1']}, {instruccion['src2']}"
        if op in {"vredsum", "vbcast"}:
            return f"{op.upper()} {instruccion['dst']}, {instruccion['src']}"
        if op in {"send", "vsend"}:
            return f"{op.upper()} {instruccion['dir']}, {instruccion['src']}"
        if op in {"recv", "vrecv"}:
            return f"{op.upper()} {instruccion['dir']}, {instruccion['dst']}"
        return op.upper()

    @staticmethod
    def indice_registro(registro):
        """Convierte un nombre (``rA``) o un número (0..15) en el índice del registro."""
        if isinstance(registro, str):
            if registro not in ISA.REGISTROS:
                raise ValueError(f"Registro desconocido: {registro}")
            return ISA.REGISTROS[registro]
        if not isinstance(registro, int) or not 0 <= registro < ISA.CANTIDAD_REGISTROS:
            raise ValueError(f"Índice de registro inválido: {registro}")
        return registro

    @staticmethod
    def indice_vectorial(registro):
        """Convierte el nombre de un registro vectorial (``v2``) en su índice."""
        if registro not in ISA.VECTORIALES:
            raise ValueError(f"Registro vectorial desconocido: {registro}")
        return ISA.VECTORIALES[registro]

    @staticmethod
    def _constante(texto):
        """La constante de un MOV convertida a float."""
        return float(texto)

    @staticmethod
    def _operando_memoria(texto, origen):
        coincidencia = PATRON_MEMORIA.fullmatch(texto)
        if not coincidencia:
            raise ValueError(f"{origen}: memoria inválida '{texto}'")
        return coincidencia.group(1), int(coincidencia.group(2))
