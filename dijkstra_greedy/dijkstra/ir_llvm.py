"""Adaptador de entrada del bloque Dijkstra (skill 04): IR de LLVM -> ``GrafoDatos``.

Los skills 02-03 dejan en ``programas/<operacion>/`` el ``operaciones.ll``
(clang -O0) del que salen los grafos ``.dot``. Los bucles de la operación
tienen límites constantes, así que aquí se **ejecuta simbólicamente** la
función ``@<operacion>``: los enteros (índices y contadores) se calculan
de verdad y cada ``load``/``store`` de float, ``fadd``... queda como un
nodo del grafo. El resultado es el grafo desenrollado, sin ciclos, que
pide el bloque.

Qué arreglo es cada región de memoria.bin se lee de
``@inicializar_<operacion>``: cada entrada de su tabla ``RegionMemoria``
guarda un nombre (``@.str.N``) y uno de sus parámetros, que en la
llamada desde ``@<operacion>`` es un ``alloca``. Los demás ``alloca``
(``acc``, contadores) son variables locales y viven en registros.

Aritmética que entiende: ``fadd``, ``fsub``, ``fmul``, ``fdiv``,
``llvm.fmuladd`` (MUL + ADD), ``fneg`` (dentro de una suma o resta se
convierte en resta: ``acc -= x * y`` queda MUL + SUB) y ``sitofp`` de un
entero conocido (una constante). Cualquier otra instrucción da ``ErrorIR``
con la línea del IR.

Dos simplificaciones de memoria, las dos exactas:

* Un ``load`` que lee lo que acaba de guardar el ``store`` anterior (sin
  otro ``store`` en medio) usa el valor directamente (``reenvio``). Así la
  suma de ``reduccion`` no hace 100 viajes de ida y vuelta a ``result``.
* Un ``store`` que otro ``store`` a la misma posición pisa sin que nadie
  lo lea se elimina.
"""

import re
from pathlib import Path

from .dfg import GrafoDatos, Operacion

LIMITE_PASOS = 20_000_000

_METADATOS = re.compile(r", (?:align \d+|![\w.]+ !\d+)")
_DEFINE = re.compile(r"^define\b[^@]*@([\w.]+)\((.*)\)")
_CADENA = re.compile(r'^@([\w.]+) = .*constant \[\d+ x i8\] c"((?:[^"\\]|\\[0-9A-Fa-f]{2})*)\\00"')
_ETIQUETA = re.compile(r"(\w+):")

_ALLOCA = re.compile(r"^%(\w+) = alloca (.+)$")
_GEP = re.compile(r"^%(\w+) = getelementptr (?:inbounds )?(.+?), ptr %(\w+)((?:, i\d+ [^,]+)+)$")
_LOAD = re.compile(r"^%(\w+) = load (float|i\d+|ptr), ptr %(\w+)$")
_STORE = re.compile(r"^store (float|i\d+|ptr) (\S+), ptr %(\w+)$")
_CAST = re.compile(r"^%(\w+) = (?:sext|zext|trunc) i\d+ (\S+) to i\d+$")
_ENTERA = re.compile(r"^%(\w+) = (add|sub|mul)(?: nsw| nuw)* i\d+ (\S+), (\S+)$")
_ICMP = re.compile(r"^%(\w+) = icmp (\w+) i\d+ (\S+), (\S+)$")
_FLOTANTE = re.compile(r"^%(\w+) = (fadd|fsub|fmul|fdiv)(?: \w+)* float (\S+), (\S+)$")
_FNEG = re.compile(r"^%(\w+) = fneg(?: \w+)* float (\S+)$")
_A_FLOTANTE = re.compile(r"^%(\w+) = [su]itofp i\d+ (\S+) to float$")
_FMULADD = re.compile(
    r"^%(\w+) = call float @llvm\.fmuladd\.f32\(float (?:noundef )?(\S+), float (?:noundef )?(\S+), "
    r"float (?:noundef )?(\S+)\)$"
)
_LLAMADA = re.compile(r"^(?:tail )?call void @([\w.]+)\((.*)\)$")
_SALTO_CONDICIONAL = re.compile(r"^br i1 %(\w+), label %(\w+), label %(\w+)$")
_SALTO = re.compile(r"^br label %(\w+)$")

_COMPARACIONES = {
    "slt": lambda a, b: a < b, "sle": lambda a, b: a <= b, "sgt": lambda a, b: a > b,
    "sge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b, "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b,
}
_ENTERAS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


class ErrorIR(ValueError):
    """El IR tiene algo que el adaptador no sabe traducir."""


class _Constante:
    """Un float constante del IR; se vuelve un nodo ``const`` la primera vez que se usa."""

    def __init__(self, valor):
        self.valor = valor
        self.nodo = None


class _Nodo:
    def __init__(self, id_operacion):
        self.id = id_operacion


class _Arreglo:
    """Un ``alloca``: una región de memoria.bin o una variable local."""

    def __init__(self, nombre, tamano, region=None):
        self.nombre = nombre
        self.tamano = tamano
        self.region = region
        self.local = {}


def _tipo(texto):
    """``[9 x [9 x float]]`` -> ``("arreglo", 9, ("arreglo", 9, "float"))``."""
    texto = texto.strip()
    coincidencia = re.fullmatch(r"\[(\d+) x (.+)\]", texto)
    if coincidencia:
        return ("arreglo", int(coincidencia.group(1)), _tipo(coincidencia.group(2)))
    if texto in ("float", "ptr") or re.fullmatch(r"i\d+", texto):
        return texto
    raise ErrorIR(f"tipo no soportado: {texto}")


def _tamano(tipo):
    """Tamaño en palabras de la memoria de la CGRA (un float, un entero o un puntero = 1)."""
    if isinstance(tipo, tuple):
        return tipo[1] * _tamano(tipo[2])
    return 1


def _float(texto):
    if texto.startswith("0x"):
        import struct
        return struct.unpack("<d", int(texto, 16).to_bytes(8, "little"))[0]
    return float(texto)


class ModuloIR:
    """Las funciones y cadenas de un ``.ll``."""

    def __init__(self, texto):
        self.funciones = {}  # nombre -> (parámetros, {etiqueta: [instrucciones]}, etiqueta inicial)
        self.cadenas = {}
        nombre = None
        for linea in texto.splitlines():
            if nombre is None:
                cadena = _CADENA.match(linea)
                if cadena:
                    self.cadenas[cadena.group(1)] = re.sub(
                        r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), cadena.group(2)
                    )
                    continue
                define = _DEFINE.match(linea)
                if define:
                    nombre = define.group(1)
                    parametros = re.findall(r"%(\w+)(?=\s*(?:,|$))", define.group(2))
                    bloques, inicial, bloque = {}, None, None
                continue
            linea = linea.split(";", 1)[0].strip()
            if linea == "}":
                self.funciones[nombre] = (parametros, bloques, inicial)
                nombre = None
            elif _ETIQUETA.fullmatch(linea):
                bloque = linea[:-1]
                bloques[bloque] = []
                inicial = inicial or bloque
            elif linea:
                if bloque is None:
                    # El primer bloque no lleva etiqueta: es %<cantidad de parámetros>.
                    bloque = inicial = str(len(parametros))
                    bloques[bloque] = []
                bloques[bloque].append(_METADATOS.sub("", linea))

    @classmethod
    def leer(cls, ruta):
        return cls(Path(ruta).read_text(encoding="utf-8"))

    def funcion(self, nombre):
        if nombre not in self.funciones:
            raise ErrorIR(f"el IR no tiene la función @{nombre}")
        return self.funciones[nombre]

    def regiones_de_inicializar(self, nombre):
        """``{índice de parámetro: nombre de región}`` de ``@inicializar_<operacion>``."""
        parametros, bloques, _ = self.funcion(nombre)
        guarda_parametro = {}  # alloca donde se guardó el parámetro -> índice
        regiones = {}
        pendiente = None
        for instrucciones in bloques.values():
            for instruccion in instrucciones:
                store = _STORE.match(instruccion)
                if store and store.group(1) == "ptr":
                    valor, destino = store.group(2), store.group(3)
                    if valor.startswith("%") and valor[1:] in parametros:
                        guarda_parametro[destino] = parametros.index(valor[1:])
                    elif valor.startswith("@") and valor[1:] in self.cadenas:
                        pendiente = self.cadenas[valor[1:]]
                    continue
                load = _LOAD.match(instruccion)
                if load and load.group(2) == "ptr" and pendiente is not None and load.group(3) in guarda_parametro:
                    regiones[guarda_parametro[load.group(3)]] = pendiente
                    pendiente = None
        if not regiones:
            raise ErrorIR(f"@{nombre} no arma ninguna tabla RegionMemoria reconocible")
        return regiones


class _Interprete:
    """Ejecuta ``@<operacion>`` y arma el grafo de datos."""

    def __init__(self, modulo, operacion, formas, reenvio=True):
        self.modulo = modulo
        self.operacion = operacion
        self.formas = formas  # región -> (filas, columnas) de memoria.bin
        self.reenvio = reenvio
        self.operaciones = []
        self.valores = {}
        self.arreglos = {}
        # Por posición de memoria: (valor, id del store, número de store, ¿lo leyó un load?)
        self.ultimo_store = {}
        self.cantidad_stores = 0
        self.muertos = set()
        self.negados = {}  # nodo de -x -> nodo de x (fneg)

    # -- grafo -------------------------------------------------------------
    def _nueva(self, op, entradas=(), **atributos):
        nueva = Operacion(id=len(self.operaciones), op=op, entradas=tuple(entradas), **atributos)
        self.operaciones.append(nueva)
        return nueva.id

    def _nodo(self, valor, origen):
        if isinstance(valor, _Nodo):
            return valor.id
        if isinstance(valor, _Constante):
            if valor.nodo is None:
                valor.nodo = self._nueva("const", valor=valor.valor, origen=origen)
            return valor.nodo
        raise ErrorIR(f"{origen}: se esperaba un float y llegó {valor!r}")

    # -- valores -----------------------------------------------------------
    def _valor(self, texto):
        if texto.startswith("%"):
            if texto[1:] not in self.valores:
                raise ErrorIR(f"@{self.operacion}: %{texto[1:]} se usa antes de definirse")
            return self.valores[texto[1:]]
        if re.fullmatch(r"-?\d+", texto):
            return int(texto)
        if texto in ("true", "false"):
            return texto == "true"
        try:
            return _Constante(_float(texto))
        except ValueError:
            raise ErrorIR(f"@{self.operacion}: operando no soportado '{texto}'") from None

    def _entero(self, texto):
        valor = self._valor(texto)
        if isinstance(valor, bool) or not isinstance(valor, int):
            raise ErrorIR(f"@{self.operacion}: '{texto}' no es un entero conocido")
        return valor

    def _puntero(self, texto):
        valor = self.valores.get(texto)
        if not (isinstance(valor, tuple) and valor[0] in self.arreglos):
            raise ErrorIR(f"@{self.operacion}: %{texto} no es un puntero a un alloca")
        return valor

    # -- memoria -----------------------------------------------------------
    def _leer(self, puntero, origen):
        arreglo = self.arreglos[puntero[0]]
        posicion = puntero[1]
        if not 0 <= posicion < arreglo.tamano:
            raise ErrorIR(f"{origen}: acceso fuera del arreglo ({posicion} de {arreglo.tamano})")
        if arreglo.region is None:
            if posicion not in arreglo.local:
                raise ErrorIR(f"{origen}: lee una variable local sin inicializar")
            return arreglo.local[posicion]
        clave = (arreglo.region, posicion)
        ultimo = self.ultimo_store.get(clave)
        if self.reenvio and ultimo is not None and ultimo[2] == self.cantidad_stores:
            return ultimo[0]
        if ultimo is not None:
            self.ultimo_store[clave] = (ultimo[0], ultimo[1], ultimo[2], True)
        return _Nodo(self._nueva("load", banco=arreglo.region, indice=posicion, origen=origen))

    def _escribir(self, puntero, valor, origen):
        arreglo = self.arreglos[puntero[0]]
        posicion = puntero[1]
        if not 0 <= posicion < arreglo.tamano:
            raise ErrorIR(f"{origen}: acceso fuera del arreglo ({posicion} de {arreglo.tamano})")
        if arreglo.region is None:
            arreglo.local[posicion] = valor
            return
        clave = (arreglo.region, posicion)
        anterior = self.ultimo_store.get(clave)
        if anterior is not None and not anterior[3]:
            self.muertos.add(anterior[1])
        nodo = self._nueva("store", [self._nodo(valor, origen)], banco=arreglo.region, indice=posicion, origen=origen)
        self.cantidad_stores += 1
        self.ultimo_store[clave] = (valor, nodo, self.cantidad_stores, False)

    # -- ejecución ---------------------------------------------------------
    def ejecutar(self):
        parametros, bloques, inicial = self.modulo.funcion(self.operacion)
        if parametros:
            raise ErrorIR(f"@{self.operacion} no debería recibir parámetros")
        regiones_parametro = None
        bloque, pasos = inicial, 0
        while True:
            siguiente = None
            for instruccion in bloques[bloque]:
                pasos += 1
                if pasos > LIMITE_PASOS:
                    raise ErrorIR(f"@{self.operacion}: más de {LIMITE_PASOS} instrucciones; ¿bucle sin fin?")
                origen = f"@{self.operacion}: {instruccion}"
                if (m := _ALLOCA.match(instruccion)):
                    tipo = _tipo(m.group(2))
                    self.arreglos[m.group(1)] = _Arreglo(m.group(1), _tamano(tipo))
                    self.valores[m.group(1)] = (m.group(1), 0)
                elif (m := _GEP.match(instruccion)):
                    tipo = _tipo(m.group(2))
                    arreglo, desplazamiento = self._puntero(m.group(3))
                    indices = [self._entero(i.split()[1]) for i in m.group(4).strip(", ").split(", ")]
                    desplazamiento += indices[0] * _tamano(tipo)
                    for indice in indices[1:]:
                        if not isinstance(tipo, tuple):
                            raise ErrorIR(f"{origen}: getelementptr dentro de un escalar")
                        tipo = tipo[2]
                        desplazamiento += indice * _tamano(tipo)
                    self.valores[m.group(1)] = (arreglo, desplazamiento)
                elif (m := _LOAD.match(instruccion)):
                    puntero = self._puntero(m.group(3))
                    if m.group(2) == "float":
                        self.valores[m.group(1)] = self._leer(puntero, origen)
                    else:
                        arreglo = self.arreglos[puntero[0]]
                        if arreglo.region is not None or puntero[1] not in arreglo.local:
                            raise ErrorIR(f"{origen}: lee un entero que no es una variable local")
                        self.valores[m.group(1)] = arreglo.local[puntero[1]]
                elif (m := _STORE.match(instruccion)):
                    puntero = self._puntero(m.group(3))
                    valor = self._valor(m.group(2)) if m.group(1) != "ptr" else self._puntero(m.group(2)[1:])
                    if m.group(1) == "float":
                        self._escribir(puntero, valor, origen)
                    else:
                        arreglo = self.arreglos[puntero[0]]
                        if arreglo.region is not None:
                            raise ErrorIR(f"{origen}: guarda un {m.group(1)} en la región '{arreglo.region}'")
                        arreglo.local[puntero[1]] = valor
                elif (m := _CAST.match(instruccion)):
                    self.valores[m.group(1)] = self._entero(m.group(2))
                elif (m := _ENTERA.match(instruccion)):
                    self.valores[m.group(1)] = _ENTERAS[m.group(2)](self._entero(m.group(3)), self._entero(m.group(4)))
                elif (m := _ICMP.match(instruccion)):
                    if m.group(2) not in _COMPARACIONES:
                        raise ErrorIR(f"{origen}: comparación no soportada")
                    self.valores[m.group(1)] = _COMPARACIONES[m.group(2)](
                        self._entero(m.group(3)), self._entero(m.group(4))
                    )
                elif (m := _FLOTANTE.match(instruccion)):
                    op, a, b = m.group(2), *(self._nodo(self._valor(m.group(i)), origen) for i in (3, 4))
                    # a + (-b) = a - b, (-a) + b = b - a y a - (-b) = a + b, exactos en IEEE.
                    if op == "fadd" and b in self.negados:
                        op, b = "fsub", self.negados[b]
                    elif op == "fadd" and a in self.negados:
                        op, a, b = "fsub", b, self.negados[a]
                    elif op == "fsub" and b in self.negados:
                        op, b = "fadd", self.negados[b]
                    self.valores[m.group(1)] = _Nodo(self._nueva(op, [a, b], origen=origen))
                elif (m := _FNEG.match(instruccion)):
                    # -x (clang la usa en acc -= x * y). El ISA no tiene NEG: -0.0 - x, que es
                    # exacto; si -x termina en una suma o resta, se resta x y este nodo se descarta.
                    valor = self._valor(m.group(2))
                    if isinstance(valor, _Constante):
                        self.valores[m.group(1)] = _Constante(-valor.valor)
                    else:
                        x = self._nodo(valor, origen)
                        negado = self._nueva("fsub", [self._nodo(_Constante(-0.0), origen), x], origen=origen)
                        self.negados[negado] = x
                        self.valores[m.group(1)] = _Nodo(negado)
                elif (m := _A_FLOTANTE.match(instruccion)):
                    # (float)i: los enteros se conocen al ejecutar, así que es una constante.
                    self.valores[m.group(1)] = _Constante(float(self._entero(m.group(2))))
                elif (m := _FMULADD.match(instruccion)):
                    # x * y + z: el skill 04 la lee como un MUL seguido de un ADD. Con
                    # x o y negado (acc -= x * y) queda z - x * y: un MUL y un SUB.
                    x, y, z = (self._nodo(self._valor(m.group(i)), origen) for i in (2, 3, 4))
                    if x in self.negados or y in self.negados:
                        producto = self._nueva("fmul", [self.negados.get(x, x), self.negados.get(y, y)], origen=origen)
                        resta = "fadd" if x in self.negados and y in self.negados else "fsub"
                        self.valores[m.group(1)] = _Nodo(self._nueva(resta, [z, producto], origen=origen))
                    else:
                        producto = self._nueva("fmul", [x, y], origen=origen)
                        self.valores[m.group(1)] = _Nodo(self._nueva("fadd", [producto, z], origen=origen))
                elif (m := _LLAMADA.match(instruccion)):
                    if m.group(1) != f"inicializar_{self.operacion}":
                        raise ErrorIR(f"{origen}: llamada no soportada")
                    regiones_parametro = self.modulo.regiones_de_inicializar(m.group(1))
                    argumentos = re.findall(r"ptr (?:noundef )?%(\w+)", m.group(2))
                    self._asignar_regiones(argumentos, regiones_parametro, origen)
                elif (m := _SALTO_CONDICIONAL.match(instruccion)):
                    condicion = self.valores.get(m.group(1))
                    if not isinstance(condicion, bool):
                        raise ErrorIR(f"{origen}: la condición no es conocida")
                    siguiente = m.group(2) if condicion else m.group(3)
                    break
                elif (m := _SALTO.match(instruccion)):
                    siguiente = m.group(1)
                    break
                elif instruccion == "ret void":
                    if regiones_parametro is None:
                        raise ErrorIR(f"@{self.operacion} no llama a @inicializar_{self.operacion}")
                    return self._grafo()
                else:
                    raise ErrorIR(f"{origen}: instrucción no soportada")
            if siguiente is None:
                raise ErrorIR(f"@{self.operacion}: el bloque %{bloque} termina sin salto")
            bloque = siguiente

    def _asignar_regiones(self, argumentos, regiones_parametro, origen):
        for indice, nombre in sorted(regiones_parametro.items()):
            if indice >= len(argumentos):
                raise ErrorIR(f"{origen}: falta el argumento {indice}")
            arreglo_id, desplazamiento = self._puntero(argumentos[indice])
            arreglo = self.arreglos[arreglo_id]
            if nombre not in self.formas:
                raise ErrorIR(f"la región '{nombre}' no está en memoria.bin")
            filas, columnas = self.formas[nombre]
            if desplazamiento != 0 or arreglo.tamano != filas * columnas:
                raise ErrorIR(
                    f"la región '{nombre}' tiene {filas * columnas} palabras y su arreglo en "
                    f"@{self.operacion} tiene {arreglo.tamano}"
                )
            arreglo.region = nombre

    def _grafo(self):
        """Quita los store pisados y lo que solo los alimentaba, y agrega las aristas de memoria."""
        vivas = set()
        pendientes = [op.id for op in self.operaciones if op.op == "store" and op.id not in self.muertos]
        while pendientes:
            actual = pendientes.pop()
            if actual in vivas:
                continue
            vivas.add(actual)
            pendientes.extend(self.operaciones[actual].entradas)
        nuevo_id = {}
        operaciones = []
        for op in self.operaciones:
            if op.id in vivas:
                nuevo_id[op.id] = len(operaciones)
                op.entradas = tuple(nuevo_id[e] for e in op.entradas)
                op.id = len(operaciones)
                operaciones.append(op)

        memoria = {op.id: [] for op in operaciones}
        ultimo_store, loads_desde = {}, {}
        for op in operaciones:
            if op.op not in ("load", "store"):
                continue
            clave = (op.banco, op.indice)
            if op.op == "load":
                if clave in ultimo_store:
                    memoria[op.id].append(ultimo_store[clave])
                loads_desde.setdefault(clave, []).append(op.id)
            else:
                if clave in ultimo_store:
                    memoria[op.id].append(ultimo_store[clave])
                memoria[op.id].extend(loads_desde.pop(clave, []))
                ultimo_store[clave] = op.id
        for op in operaciones:
            op.memoria = tuple(memoria[op.id])
        return GrafoDatos(self.operacion, operaciones, dict(self.formas))


def grafo_desde_ir(ruta_ll, operacion, formas, reenvio=True):
    """Grafo de datos desenrollado de ``@<operacion>``.

    ``formas`` son las regiones de memoria.bin: ``{nombre: (filas, columnas)}``.
    """
    return _Interprete(ModuloIR.leer(ruta_ll), operacion, formas, reenvio).ejecutar()
