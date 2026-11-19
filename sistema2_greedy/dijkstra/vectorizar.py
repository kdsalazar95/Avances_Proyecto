"""Vectorización del grafo de datos (parte del skill 04/05 en modo vectorial).

Agrupa ``ancho`` (``ancho_vector`` de cgra.json) cálculos escalares
idénticos en uno vectorial, al estilo SLP: se parte de grupos de
``ancho`` stores a posiciones seguidas del mismo banco y se comparan los
árboles de los valores que guardan, carril por carril:

* ``load`` a posiciones seguidas            -> ``vload`` (VLD)
* el mismo valor (o el mismo ``load``/``const``) en todos los carriles
                                            -> ``vbcast`` (VBCAST) de ese escalar
* la misma aritmética en todos los carriles -> ``vadd``/``vsub``/``vmul``/``vdiv``

Así ``c[i:i+4] = a[i:i+4] + b[i:i+4]`` (reduccion) y
``result[i][j:j+4] += a[i][k] * b[k][j:j+4]`` (matmul, convolucion) quedan
vectoriales. Además, una cadena de sumas ``acc = acc + x`` (la suma de
``reduccion`` y las columnas que sobran de matmul/convolucion) se agrupa
de a ``ancho``: ``acc = acc + VREDSUM(vector de las x)`` cuando las ``x``
se pueden vectorizar.

Lo que no encaja (carriles distintos, accesos no contiguos, como las
columnas de ``b`` en matmul) queda escalar. Un grupo solo se acepta si los
nodos que reemplaza no se usan fuera de él y si mover sus accesos a
memoria no cruza un store a esas posiciones.

La suma con VREDSUM cambia el orden de las sumas respecto al C, así que en
float el redondeo puede ser distinto (con los datos enteros del proyecto
el resultado es idéntico).
"""

import heapq

from .dfg import GrafoDatos, Operacion

VECTORIAL = {"fadd": "vadd", "fsub": "vsub", "fmul": "vmul", "fdiv": "vdiv"}
PRODUCEN_VECTOR = frozenset({"vload", "vadd", "vsub", "vmul", "vdiv", "vbcast"})
ACCESOS = frozenset({"load", "store", "vload", "vstore"})


def es_vector(op):
    """¿La operación produce un vector (va a un registro v0..v7)?"""
    return op in PRODUCEN_VECTOR


class _Vectorizador:
    def __init__(self, grafo, ancho):
        self.grafo = grafo
        self.ancho = ancho
        self.ops = grafo.operaciones
        self.consumidores = grafo.consumidores()
        self.absorbidos = set()  # escalares reemplazados por vectores
        self.reemplazo = {}  # escalar absorbido -> referencia que lo sustituye
        self.nuevos = []  # [op, entradas (referencias), atributos, clave]
        self.usos_nuevos = {}  # escalar -> cuántos nodos nuevos lo leen
        # Accesos a memoria por banco, para no reordenar sobre un store.
        self.accesos = {}
        for op in self.ops:
            if op.op in ("load", "store"):
                self.accesos.setdefault(op.banco, []).append(op)

    # -- referencias: ("s", id original) o ("n", índice en nuevos) --------------
    def _nuevo(self, op, entradas, clave, **atributos):
        self.nuevos.append((op, entradas, atributos, clave))
        return ("n", len(self.nuevos) - 1)

    def _sin_conflicto(self, banco, inicio, fin, desde, hasta, contra):
        """¿Ningún acceso de tipo ``contra`` a ``banco[inicio:fin]`` entre los ids ``desde`` y ``hasta``?"""
        for op in self.accesos.get(banco, ()):
            if desde < op.id < hasta and op.op in contra and inicio <= op.indice < fin:
                return False
        return True

    def _combinar(self, carriles, absorbe, memo):
        """Referencia vectorial que vale ``carriles`` (un id escalar por carril), o ``None``."""
        if carriles in memo:
            return memo[carriles]
        memo[carriles] = None
        ops = [self.ops[i] for i in carriles]
        primero = ops[0]
        resultado = None
        if all(i == carriles[0] for i in carriles):
            resultado = self._nuevo("vbcast", [("s", carriles[0])], carriles[0] + 0.5)
        elif all(o.op == "load" and o.banco == primero.banco for o in ops):
            indices = [o.indice for o in ops]
            desde, hasta = min(carriles), max(carriles)
            if indices == list(range(indices[0], indices[0] + self.ancho)):
                fin = indices[0] + self.ancho
                if self._sin_conflicto(primero.banco, indices[0], fin, desde, hasta, ("store",)):
                    absorbe.update(carriles)
                    resultado = self._nuevo("vload", [], hasta, banco=primero.banco, indice=indices[0],
                                            ancho=self.ancho)
            elif len(set(indices)) == 1 and self._sin_conflicto(
                    primero.banco, indices[0], indices[0] + 1, desde, hasta, ("store",)):
                absorbe.update(carriles[1:])
                resultado = self._nuevo("vbcast", [("s", carriles[0])], carriles[0] + 0.5)
        elif all(o.op == "const" and o.valor == primero.valor for o in ops):
            absorbe.update(carriles[1:])
            resultado = self._nuevo("vbcast", [("s", carriles[0])], carriles[0] + 0.5)
        elif primero.op in VECTORIAL and all(o.op == primero.op and len(o.entradas) == 2 for o in ops):
            hijos = [self._combinar(tuple(o.entradas[p] for o in ops), absorbe, memo) for p in range(2)]
            if None not in hijos:
                absorbe.update(carriles)
                resultado = self._nuevo(VECTORIAL[primero.op], hijos, max(carriles))
        memo[carriles] = resultado
        return resultado

    def _aceptar(self, absorbe, permitidos, inicio_nuevos):
        """Acepta el grupo si sus escalares absorbidos no se usan fuera de él
        (``permitidos``: consumidores que se redirigen al nodo nuevo)."""
        for i in absorbe:
            if i in self.absorbidos or self.usos_nuevos.get(i):
                return False
            if any(c not in absorbe and c not in permitidos.get(i, ()) for c in self.consumidores[i]):
                return False
        for _, entradas, _, _ in self.nuevos[inicio_nuevos:]:
            for tipo, i in entradas:
                if tipo == "s" and (i in absorbe or i in self.absorbidos):
                    return False
        for _, entradas, _, _ in self.nuevos[inicio_nuevos:]:
            for tipo, i in entradas:
                if tipo == "s":
                    self.usos_nuevos[i] = self.usos_nuevos.get(i, 0) + 1
        self.absorbidos |= absorbe
        return True

    def _intentar_grupo(self, raices, absorbe, permitidos, construir):
        inicio = len(self.nuevos)
        vector = self._combinar(raices, absorbe, {})
        if vector is not None:
            construir(vector)
            if self._aceptar(absorbe, permitidos, inicio):
                return True
        del self.nuevos[inicio:]
        return False

    def _agrupar_stores(self):
        por_posicion = {}
        for op in self.ops:
            if op.op == "store":
                por_posicion.setdefault((op.banco, op.indice), []).append(op)
        for op in self.ops:
            if op.op != "store" or op.id in self.absorbidos:
                continue
            carriles = []
            for desplazamiento in range(self.ancho):
                stores = por_posicion.get((op.banco, op.indice + desplazamiento), [])
                if len(stores) != 1 or stores[0].id in self.absorbidos:
                    break
                carriles.append(stores[0])
            if len(carriles) < self.ancho:
                continue
            ids = tuple(s.id for s in carriles)
            desde, hasta = min(ids), max(ids)
            if not self._sin_conflicto(op.banco, op.indice, op.indice + self.ancho, desde, hasta, ("load",)):
                continue
            absorbe = set(ids)
            self._intentar_grupo(
                tuple(s.entradas[0] for s in carriles), absorbe, {},
                lambda vector: self._nuevo("vstore", [vector], hasta, banco=op.banco, indice=op.indice,
                                           ancho=self.ancho),
            )

    def _cadenas(self):
        """Cadenas ``n_k = fadd(n_{k-1}, x_k)``: [(nodo, posición del operando de la cadena)]."""
        def eslabon(op):
            if op.op != "fadd" or op.id in self.absorbidos:
                return None
            posiciones = [p for p, e in enumerate(op.entradas)
                          if self.ops[e].op == "fadd" and e not in self.absorbidos
                          and self.consumidores[e] == [op.id] and op.entradas.count(e) == 1]
            return posiciones[0] if len(posiciones) == 1 else None

        enlaces = {op.id: p for op in self.ops if (p := eslabon(op)) is not None}
        cadenas, en_cadena = [], {}
        for op in self.ops:
            if op.id not in enlaces:
                continue
            previo = op.entradas[enlaces[op.id]]
            if previo in en_cadena:
                cadena = en_cadena[previo]
            else:
                cadena = []
                cadenas.append(cadena)
                # El primer eslabón no tiene una suma antes: se reconoce por la posición del siguiente.
                anterior = self.ops[previo]
                if (anterior.op == "fadd" and previo not in enlaces and previo not in self.absorbidos
                        and self.consumidores[previo] == [op.id]):
                    cadena.append((previo, enlaces[op.id]))
            cadena.append((op.id, enlaces[op.id]))
            en_cadena[op.id] = cadena
        return cadenas

    def _referencia(self, i):
        ref = ("s", i)
        while ref[0] == "s" and ref[1] in self.reemplazo:
            ref = self.reemplazo[ref[1]]
        return ref

    def _agrupar_cadenas(self):
        for cadena in self._cadenas():
            k = 0
            while k + self.ancho <= len(cadena):
                grupo = cadena[k:k + self.ancho]
                ids = [n for n, _ in grupo]
                primero, posicion = self.ops[ids[0]], grupo[0][1]
                xs = tuple(self.ops[n].entradas[1 - p] for n, p in grupo)
                previo = self._referencia(primero.entradas[posicion])
                ultimo = ids[-1]
                absorbe = set(ids)

                def construir(vector, previo=previo, posicion=posicion, ultimo=ultimo):
                    suma = self._nuevo("vredsum", [vector], ultimo - 0.25)
                    entradas = [previo, suma] if posicion == 0 else [suma, previo]
                    self.reemplazo[ultimo] = self._nuevo("fadd", entradas, ultimo)

                if self._intentar_grupo(xs, absorbe, {ultimo: set(self.consumidores[ultimo])}, construir):
                    k += self.ancho
                else:
                    self.reemplazo.pop(ultimo, None)
                    k += 1

    def grafo_final(self):
        self._agrupar_stores()
        self._agrupar_cadenas()
        if not self.nuevos:
            return None
        # Nodos: originales que quedan y nuevos, ordenados por clave respetando dependencias.
        nodos = {}
        for op in self.ops:
            if op.id not in self.absorbidos:
                entradas = [self._referencia(e) for e in op.entradas]
                atributos = dict(banco=op.banco, indice=op.indice, valor=op.valor, origen=op.origen)
                nodos[("s", op.id)] = (op.op, entradas, atributos, float(op.id))
        for k, (op, entradas, atributos, clave) in enumerate(self.nuevos):
            nodos[("n", k)] = (op, [self._referencia(e[1]) if e[0] == "s" else e for e in entradas],
                               atributos, clave)
        # Un vbcast va justo antes de su primer consumidor: ubicarlo antes solo
        # ocuparía un registro vectorial esperando.
        primer_uso = {}
        for _, entradas, _, clave in nodos.values():
            for entrada in entradas:
                entrada = tuple(entrada)
                primer_uso[entrada] = min(primer_uso.get(entrada, clave), clave)
        for ref, (op, entradas, atributos, clave) in list(nodos.items()):
            if op == "vbcast" and ref in primer_uso:
                nodos[ref] = (op, entradas, atributos, primer_uso[ref] - 0.5)
        faltan = {ref: 0 for ref in nodos}
        sucesores = {ref: [] for ref in nodos}
        for ref, (_, entradas, _, _) in nodos.items():
            for entrada in dict.fromkeys(map(tuple, entradas)):
                if entrada not in nodos:
                    raise RuntimeError(f"vectorización: {ref} usa {entrada}, que no existe")
                faltan[ref] += 1
                sucesores[entrada].append(ref)
        listos = [(nodos[ref][3], ref) for ref, n in faltan.items() if n == 0]
        heapq.heapify(listos)
        nuevo_id, operaciones = {}, []
        while listos:
            _, ref = heapq.heappop(listos)
            op, entradas, atributos, _ = nodos[ref]
            nuevo_id[ref] = len(operaciones)
            operaciones.append(Operacion(id=len(operaciones), op=op,
                                         entradas=tuple(nuevo_id[tuple(e)] for e in entradas), **atributos))
            for siguiente in sucesores[ref]:
                faltan[siguiente] -= 1
                if faltan[siguiente] == 0:
                    heapq.heappush(listos, (nodos[siguiente][3], siguiente))
        if len(operaciones) != len(nodos):
            raise RuntimeError("vectorización: el grafo resultante tiene ciclos")
        _aristas_de_memoria(operaciones)
        return GrafoDatos(self.grafo.nombre, operaciones, dict(self.grafo.regiones))


def _aristas_de_memoria(operaciones):
    """Ordena los accesos que se solapan (lectura después de escritura, etc.)."""
    stores, loads = {}, {}
    for op in operaciones:
        if op.op not in ACCESOS:
            continue
        inicio, fin = op.indice, op.indice + (op.ancho or 1)
        es_store = op.op in ("store", "vstore")
        previas = [s.id for s in stores.get(op.banco, ()) if s.indice < fin and inicio < s.indice + (s.ancho or 1)]
        if es_store:
            previas += [l.id for l in loads.get(op.banco, ()) if l.indice < fin and inicio < l.indice + (l.ancho or 1)]
            stores.setdefault(op.banco, []).append(op)
        else:
            loads.setdefault(op.banco, []).append(op)
        op.memoria = tuple(sorted(set(previas)))


def vectorizar(grafo, ancho):
    """Grafo con las operaciones agrupadas de a ``ancho``; el mismo grafo si nada se vectoriza."""
    if not ancho or ancho < 2:
        return grafo
    vectorizado = _Vectorizador(grafo, ancho).grafo_final()
    return vectorizado if vectorizado is not None else grafo
