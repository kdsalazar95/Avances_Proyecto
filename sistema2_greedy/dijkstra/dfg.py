"""Grafo de datos (DFG) que recibe el bloque Dijkstra (salida del skill 04).

Cada ``Operacion`` es una instrucción ya desenrollada: un ``load`` o
``store`` a una posición concreta de una región de memoria.bin, una
aritmética (``fadd``, ``fsub``, ``fmul``, ``fdiv``) o una constante
(``const``). En modo vectorial (``vectorizar.py``) aparecen también
``vload``/``vstore`` de ``ancho`` posiciones seguidas, ``vadd``... carril a
carril, ``vbcast`` (un escalar a todos los carriles) y ``vredsum`` (la
suma de los carriles). Las aristas de datos van del productor al consumidor, en el
orden de los operandos; las de memoria solo ordenan accesos a la misma
posición (un ``load`` después del ``store`` que lee, etc.).
"""

import heapq
from dataclasses import dataclass, field


@dataclass
class Operacion:
    """Un nodo del grafo de datos."""

    id: int
    op: str
    entradas: tuple = ()  # productores de los operandos, en orden
    memoria: tuple = ()  # accesos a la misma posición que deben ir antes
    banco: str = None  # load/store: región de memoria.bin
    indice: int = None  # load/store: posición dentro de la región
    ancho: int = None  # vload/vstore: cuántas posiciones seguidas
    valor: float = None  # const
    origen: str = ""  # de dónde salió (por ejemplo, la instrucción del IR)

    def texto(self):
        if self.op in ("load", "store"):
            return f"{self.op} {self.banco}[{self.indice}]"
        if self.op in ("vload", "vstore"):
            return f"{self.op} {self.banco}[{self.indice}..{self.indice + self.ancho - 1}]"
        if self.op == "const":
            return f"const {self.valor}"
        return f"{self.op}({', '.join(f'#{e}' for e in self.entradas)})"


@dataclass
class GrafoDatos:
    """Operaciones de una función, en el orden en que las ejecuta el programa."""

    nombre: str
    operaciones: list
    regiones: dict = field(default_factory=dict)  # nombre -> (filas, columnas)
    vectorizados: dict = field(default_factory=dict, repr=False, compare=False)  # ancho -> grafo

    def __len__(self):
        return len(self.operaciones)

    def operacion(self, id_operacion):
        return self.operaciones[id_operacion]

    def consumidores(self):
        """``{id: [consumidores de su valor]}``, cada consumidor una sola vez."""
        consumidores = {op.id: [] for op in self.operaciones}
        for op in self.operaciones:
            for entrada in dict.fromkeys(op.entradas):
                consumidores[entrada].append(op.id)
        return consumidores

    def validar(self):
        """Ids consecutivos y aristas hacia operaciones que existen."""
        for posicion, op in enumerate(self.operaciones):
            if op.id != posicion:
                raise ValueError(f"la operación en la posición {posicion} tiene id {op.id}")
            for otra in op.entradas + op.memoria:
                if not 0 <= otra < len(self.operaciones):
                    raise ValueError(f"la operación #{op.id} depende de #{otra}, que no existe")

    def orden_topologico(self):
        """Orden topológico (datos y memoria) que respeta el orden del programa.

        Entre las operaciones listas se toma siempre la de menor id, así que
        el orden es determinístico. Si el grafo tiene ciclos, ``ValueError``.
        """
        faltan = [0] * len(self.operaciones)
        sucesores = [[] for _ in self.operaciones]
        for op in self.operaciones:
            for previa in dict.fromkeys(op.entradas + op.memoria):
                faltan[op.id] += 1
                sucesores[previa].append(op.id)
        listas = [op.id for op in self.operaciones if faltan[op.id] == 0]
        heapq.heapify(listas)
        orden = []
        while listas:
            actual = heapq.heappop(listas)
            orden.append(actual)
            for siguiente in sucesores[actual]:
                faltan[siguiente] -= 1
                if faltan[siguiente] == 0:
                    heapq.heappush(listas, siguiente)
        if len(orden) != len(self.operaciones):
            en_ciclo = [op.id for op in self.operaciones if faltan[op.id] > 0]
            raise ValueError(f"el grafo tiene ciclos (operaciones {en_ciclo[:10]})")
        return orden

    @classmethod
    def desde_dict(cls, datos):
        """Arma el grafo desde un diccionario, útil para pruebas y otros adaptadores::

            {"nombre": "ejemplo", "regiones": {"a": [1, 4]},
             "operaciones": [{"id": 0, "op": "load", "banco": "a", "indice": 0},
                             {"id": 1, "op": "fadd", "entradas": [0, 0]}]}
        """
        operaciones = [
            Operacion(
                id=o["id"], op=o["op"], entradas=tuple(o.get("entradas", ())), memoria=tuple(o.get("memoria", ())),
                banco=o.get("banco"), indice=o.get("indice"), ancho=o.get("ancho"), valor=o.get("valor"),
                origen=o.get("origen", ""),
            )
            for o in datos["operaciones"]
        ]
        regiones = {nombre: tuple(forma) for nombre, forma in datos.get("regiones", {}).items()}
        return cls(datos.get("nombre", "grafo"), operaciones, regiones)


def evaluar(grafo, regiones):
    """Ejecuta el grafo en orden de programa sobre ``{región: [valores]}``.

    Devuelve las regiones finales. La aritmética es la del simulador
    (floats de Python, ``fmuladd`` como MUL y luego ADD), así que el
    resultado se puede comparar exactamente con la memoria de la CGRA.
    Los vectores son tuplas, y ``vredsum`` suma con ``sum()`` como el simulador.
    """
    memoria = {nombre: list(valores) for nombre, valores in regiones.items()}
    valores = {}
    anchos = _anchos(grafo)
    aritmetica = {
        "fadd": lambda a, b: a + b, "fsub": lambda a, b: a - b,
        "fmul": lambda a, b: a * b, "fdiv": lambda a, b: a / b,
    }
    for op in grafo.operaciones:
        if op.op == "load":
            valores[op.id] = memoria[op.banco][op.indice]
        elif op.op == "store":
            memoria[op.banco][op.indice] = valores[op.entradas[0]]
        elif op.op == "vload":
            valores[op.id] = tuple(memoria[op.banco][op.indice:op.indice + op.ancho])
        elif op.op == "vstore":
            memoria[op.banco][op.indice:op.indice + op.ancho] = list(valores[op.entradas[0]])
        elif op.op == "const":
            valores[op.id] = op.valor
        elif op.op == "vbcast":
            valores[op.id] = (valores[op.entradas[0]],) * anchos[op.id]
        elif op.op == "vredsum":
            valores[op.id] = sum(valores[op.entradas[0]])
        elif op.op.startswith("v"):
            a, b = (valores[e] for e in op.entradas)
            valores[op.id] = tuple(map(aritmetica["f" + op.op[1:]], a, b))
        else:
            valores[op.id] = aritmetica[op.op](*(valores[e] for e in op.entradas))
    return memoria


def _anchos(grafo):
    """Ancho de cada vector: el de los vload/vstore conectados (todos iguales en un grafo)."""
    anchos = {op.ancho for op in grafo.operaciones if op.ancho}
    ancho = anchos.pop() if len(anchos) == 1 else None
    return {op.id: ancho for op in grafo.operaciones if op.op == "vbcast"}
