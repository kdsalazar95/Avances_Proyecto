"""Formato de la CGRA: dimensiones, memoria, ancho de los vectores y tipos de PE en JSON."""

import json
from pathlib import Path

from isa import ISA
from malla import Malla

RUTA_POR_DEFECTO = Path(__file__).with_name("cgra.json")
BORDES = ("norte", "sur", "oeste", "este")
PARIDADES = ("filas_pares", "filas_impares")  # la fila 0 es par
CICLO = "ciclo_filas"
ADAPTAR = "adaptar_mallas_pequenas"  # true: malla_pequena() reemplaza a 'distribucion' en mallas chicas
INSTRUCCIONES_ISA = ISA.INSTRUCCIONES


class FormatoCGRA:
    """Descripción de la malla, memoria y capacidades de los PEs.

    Si no se especifican tipos, todos los PEs reciben el ISA completo y la
    CGRA conserva el comportamiento homogéneo (sin las instrucciones
    vectoriales si no hay ``ancho_vector``).

    ``ancho_vector`` es el número de palabras de cada registro vectorial,
    igual para toda la CGRA. Solo hace falta si algún tipo de PE tiene
    instrucciones vectoriales.
    """

    CAMPOS = ("filas", "columnas", "palabras_memoria")

    def __init__(self, filas, columnas, palabras_memoria, tipos_pe=None, malla=None, ancho_vector=None):
        if type(filas) is not int or type(columnas) is not int or filas <= 0 or columnas <= 0:
            raise ValueError("La malla debe tener al menos una fila y una columna")
        if type(palabras_memoria) is not int or palabras_memoria <= 0:
            raise ValueError("La memoria de la CGRA debe tener al menos una palabra")
        if ancho_vector is not None and (type(ancho_vector) is not int or ancho_vector <= 0):
            raise ValueError("'ancho_vector' debe ser un entero mayor que cero")

        self.filas = filas
        self.columnas = columnas
        self.palabras_memoria = palabras_memoria
        self.ancho_vector = ancho_vector
        if tipos_pe is None:
            completo = ISA.INSTRUCCIONES if ancho_vector else ISA.INSTRUCCIONES_ESCALARES
            tipos_pe = {"G": {"nombre": "generico", "descripcion": "", "instrucciones": completo}}
        if malla is None:
            if len(tipos_pe) != 1:
                raise ValueError("Debe indicar una malla o distribución para asignar los tipos de PE")
            malla = [[next(iter(tipos_pe)) for _ in range(columnas)] for _ in range(filas)]

        self.tipos_pe = {
            letra: {
                "nombre": tipo.get("nombre", letra),
                "descripcion": tipo.get("descripcion", ""),
                "instrucciones": tuple(op.upper() for op in tipo["instrucciones"]),
            }
            for letra, tipo in tipos_pe.items()
        }
        self.malla = [list(fila) for fila in malla]
        self._validar()

    def _validar(self):
        if not self.tipos_pe:
            raise ValueError("Falta definir al menos un tipo de PE en 'tipos_pe'")
        for letra, tipo in self.tipos_pe.items():
            if len(letra) != 1 or not letra.isalpha():
                raise ValueError(f"El tipo '{letra}' debe ser una sola letra")
            desconocidas = set(tipo["instrucciones"]) - set(INSTRUCCIONES_ISA)
            if desconocidas:
                raise ValueError(
                    f"El tipo '{letra}' usa instrucciones que no existen: {', '.join(sorted(desconocidas))}"
                )
            vectoriales = set(tipo["instrucciones"]) & set(ISA.INSTRUCCIONES_VECTORIALES)
            if vectoriales and not self.ancho_vector:
                raise ValueError(
                    f"El tipo '{letra}' usa instrucciones vectoriales ({', '.join(sorted(vectoriales))}) "
                    "y falta 'ancho_vector'"
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
        """Letra del tipo asignado a un PE."""
        fila, columna = Malla.posicion(pe_id)
        return self.malla[fila][columna]

    def nombre_tipo(self, pe_id):
        return self.tipos_pe[self.letra(pe_id)]["nombre"]

    def instrucciones(self, pe_id):
        return {op.lower() for op in self.tipos_pe[self.letra(pe_id)]["instrucciones"]}

    def mapa(self):
        return [" ".join(fila) for fila in self.malla]

    def leyenda(self):
        lineas = [
            f"{letra} = {tipo['nombre']}: {', '.join(tipo['instrucciones'])}"
            for letra, tipo in self.tipos_pe.items()
        ]
        if self.ancho_vector:
            lineas.append(f"Vectores de {self.ancho_vector} palabras (v0..v{ISA.CANTIDAD_VECTORIALES - 1})")
        return lineas

    @classmethod
    def cargar(cls, ruta=RUTA_POR_DEFECTO, filas=None, columnas=None):
        """Lee y valida el JSON con el formato de la CGRA.

        ``filas`` y ``columnas`` del JSON pueden ser un número (una sola malla)
        o un rango ``{"minimo": a, "maximo": b}``. Con un rango, el tamaño de la
        corrida se pasa en ``filas``/``columnas`` y debe estar dentro de él; con
        un número, si se pasa un tamaño debe ser ese mismo número.
        """
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
        try:
            datos["filas"] = elegir_tamano("filas", datos["filas"], filas)
            datos["columnas"] = elegir_tamano("columnas", datos["columnas"], columnas)
        except ValueError as error:
            raise ValueError(f"{ruta}: {error}") from None
        no_enteros = [campo for campo in cls.CAMPOS if type(datos[campo]) is not int]
        if no_enteros:
            raise ValueError(f"{ruta}: {', '.join(no_enteros)} deben ser números enteros")

        try:
            ancho_vector = datos.get("ancho_vector")
            if "tipos_pe" not in datos:
                return cls(datos["filas"], datos["columnas"], datos["palabras_memoria"], ancho_vector=ancho_vector)
            tipos_pe = {
                letra: {
                    "nombre": tipo.get("nombre", letra),
                    "descripcion": tipo.get("descripcion", ""),
                    "instrucciones": tipo["instrucciones"],
                }
                for letra, tipo in datos["tipos_pe"].items()
            }
            adaptar = datos.get(ADAPTAR, False)
            if type(adaptar) is not bool:
                raise ValueError(f"'{ADAPTAR}' debe ser true o false")
            pequena = malla_pequena(datos["filas"], datos["columnas"]) if adaptar and "malla" not in datos else None
            if "malla" in datos:
                malla = [fila.split() if isinstance(fila, str) else list(fila) for fila in datos["malla"]]
            elif pequena is not None:
                malla = pequena
            elif "distribucion" in datos:
                malla = generar_malla(datos["filas"], datos["columnas"], datos["distribucion"])
            else:
                raise ValueError("falta 'malla' o 'distribucion' para asignar los tipos de PE")
            return cls(datos["filas"], datos["columnas"], datos["palabras_memoria"], tipos_pe, malla, ancho_vector)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{ruta}: {error}") from None


def elegir_tamano(eje, declarado, pedido):
    """Tamaño de un eje (``filas`` o ``columnas``) para esta corrida.

    ``declarado`` es lo que dice el JSON: un número o un rango
    ``{"minimo", "maximo"}``. ``pedido`` es el tamaño elegido por quien corre
    la CGRA (None si no eligió). Devuelve el número a usar.
    """
    if pedido is not None and (type(pedido) is not int or pedido <= 0):
        raise ValueError(f"el tamaño pedido de {eje} debe ser un entero mayor que cero")
    if isinstance(declarado, dict):
        minimo, maximo = declarado.get("minimo", 1), declarado.get("maximo")
        if type(minimo) is not int or minimo <= 0 or (maximo is not None and (type(maximo) is not int or maximo < minimo)):
            raise ValueError(f"el rango de '{eje}' debe tener 'minimo' >= 1 y 'maximo' >= 'minimo' (enteros)")
        if pedido is None:
            hasta = maximo if maximo is not None else "..."
            raise ValueError(
                f"'{eje}' es un rango [{minimo}, {hasta}]: falta elegir el tamaño (--{eje} en run_cgra.py)"
            )
        if pedido < minimo or (maximo is not None and pedido > maximo):
            raise ValueError(f"{eje} = {pedido} está fuera del rango [{minimo}, {maximo}]")
        return pedido
    if pedido is not None and type(declarado) is int and pedido != declarado:
        raise ValueError(f"este formato describe una sola malla con {eje} = {declarado}, no {pedido}")
    return declarado


def malla_pequena(filas, columnas):
    """Malla de una CGRA que no admite la 'distribucion' de bordes, o None si la admite.

    Las filas de cómputo rotan escalar (E) y vectorial (V). Una malla con una
    sola fila de cómputo no alcanza para poner E y V, y no es válida:

        1x1, 2x1, 1x2, 1 fila        no válida (ValueError)
        Nx1, N >= 3                  M arriba, E, V, E... debajo
        Nx2, N >= 2                  M E, M V, M E...
        2+ filas, 3+ columnas        None: se usa la 'distribucion' (M E..E M, M V..V M...)
    """
    def computo(cantidad_filas):
        """Tipo de cómputo de cada fila: E, V, E..."""
        if cantidad_filas < 2:
            raise ValueError(
                f"la malla {filas}x{columnas} no es válida: tiene una sola fila de cómputo y no alcanza "
                "para poner escalar (E) y vectorial (V)"
            )
        return ["E" if f % 2 == 0 else "V" for f in range(cantidad_filas)]

    if columnas == 1:
        return [["M"]] + [[letra] for letra in computo(filas - 1)]
    if columnas == 2 or filas == 1:
        letras = computo(filas)  # con una sola fila falla
        return [["M", letra] for letra in letras]
    return None


def generar_malla(filas, columnas, distribucion):
    """Genera una cuadrícula de tipos usando reglas para bordes y resto.

    ``distribucion`` es una regla (``{"oeste": "M", "este": "M", "resto": "E"}``),
    una regla por paridad de fila, para alternar filas de tipos distintos:

        {"filas_pares":   {"oeste": "M", "este": "M", "resto": "E"},
         "filas_impares": {"oeste": "M", "este": "M", "resto": "V"}}

    o un ciclo de reglas que se repite fila por fila (la fila f usa la regla
    f mod la cantidad de reglas):

        {"ciclo_filas": [{"oeste": "M", "este": "M", "resto": "E"},
                         {"oeste": "M", "este": "M", "resto": "V"}]}
    """
    if CICLO in distribucion:
        if len(distribucion) != 1:
            raise ValueError(f"'distribucion' mezcla {CICLO} con otras claves")
        reglas = distribucion[CICLO]
        if not isinstance(reglas, list) or not reglas:
            raise ValueError(f"'{CICLO}' debe ser una lista con al menos una regla")
        por_regla = [generar_malla(filas, columnas, regla) for regla in reglas]
        return [por_regla[fila % len(reglas)][fila] for fila in range(filas)]

    if set(distribucion) & set(PARIDADES):
        desconocidas = set(distribucion) - set(PARIDADES)
        if desconocidas:
            raise ValueError(
                f"'distribucion' mezcla {', '.join(PARIDADES)} con otras claves: {', '.join(sorted(desconocidas))}"
            )
        faltan = [clave for clave in PARIDADES if clave not in distribucion]
        if faltan:
            raise ValueError(f"'distribucion' necesita {', '.join(faltan)}")
        por_paridad = [generar_malla(filas, columnas, distribucion[clave]) for clave in PARIDADES]
        return [por_paridad[fila % 2][fila] for fila in range(filas)]

    desconocidas = set(distribucion) - set(BORDES) - {"resto"}
    if desconocidas:
        raise ValueError(f"'distribucion' tiene claves desconocidas: {', '.join(sorted(desconocidas))}")
    if "resto" not in distribucion:
        raise ValueError("'distribucion' necesita 'resto'")

    def tipo(fila, columna):
        if columna == 0 and "oeste" in distribucion:
            return distribucion["oeste"]
        if columna == columnas - 1 and "este" in distribucion:
            return distribucion["este"]
        if fila == 0 and "norte" in distribucion:
            return distribucion["norte"]
        if fila == filas - 1 and "sur" in distribucion:
            return distribucion["sur"]
        return distribucion["resto"]

    return [[tipo(fila, columna) for columna in range(columnas)] for fila in range(filas)]


def validar_tipos(programa, formato):
    """Rechaza instrucciones que un PE no puede ejecutar por su tipo."""
    errores = []
    for pe_id, instrucciones in programa.instrucciones.items():
        permitidas = formato.instrucciones(pe_id)
        for ciclo, instruccion in sorted(instrucciones.items()):
            op = instruccion["op"]
            if op not in permitidas:
                errores.append(
                    f"{pe_id} ({formato.nombre_tipo(pe_id)}), ciclo {ciclo}: {op.upper()} no está permitido"
                )
    if errores:
        detalle = "\n  ".join(errores[:10])
        extra = f"\n  ... y {len(errores) - 10} más" if len(errores) > 10 else ""
        raise ValueError(f"{len(errores)} instrucciones no respetan el tipo de su PE:\n  {detalle}{extra}")
