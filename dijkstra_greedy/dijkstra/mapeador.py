"""Núcleo del bloque Dijkstra (skill 07): ubica cada operación en un PE y un
ciclo, y rutea sus operandos por la malla extendida en el tiempo.

Modelo del hardware (el del simulador, ``cgra/src/malla.py`` y ``cgra/src/programa.py``):

* Cada instrucción dura 1 ciclo y cada PE ejecuta una por ciclo.
* Un dato pasa a un vecino (N, S, O, E) con ``SEND`` en el emisor y
  ``RECV`` en el receptor **en el mismo ciclo**: ese ciclo ocupa a los dos
  PEs, y el dato se puede usar en el receptor desde el ciclo siguiente.
  Un PE intermedio reenvía con ``RECV`` y luego ``SEND``.
* Cada PE tiene ``len(malla.registros)`` registros escalares (5 en el ISA:
  ``rA rB rC rT acc``) y ``len(malla.registros_vectoriales)`` vectoriales
  (``v0``..``v7``). Un valor ocupa un registro de su clase desde el ciclo en
  que se escribe hasta el último en que se lee; esperar en un PE gasta
  registro. Los vectores viajan con ``VSEND``/``VRECV``, así que solo pasan
  por PEs que tienen esas instrucciones.

Nodo del grafo en el tiempo: ``(PE, ciclo)`` = "el valor se puede leer en
ese PE en ese ciclo". Aristas: **esperar** ``(p, t) -> (p, t+1)`` (un
registro en ``p``) y **reenviar** ``(q, t) -> (p, t+1)`` (``q`` y ``p`` libres
en ``t``). Para cada operando, Dijkstra busca **hacia atrás** desde
``(PE destino, ciclo de ejecución)`` con la clave ``(-ciclo, saltos, PE)``:
la primera fuente que encuentra es la que sale más tarde y, a igual
ciclo, con menos saltos. Salir lo más tarde posible deja los valores el
menor tiempo en registros.

Los ``load`` y ``const`` no tienen operandos: se ubican junto con su
primer consumidor (la búsqueda elige también su PE y su ciclo), para que
no se carguen cientos de ciclos antes de usarse.
"""

import heapq
import multiprocessing
from dataclasses import dataclass, field

from .vectorizar import es_vector

PEREZOSAS = ("load", "vload", "const")


@dataclass
class Opciones:
    """Opciones del modelo.

    ``reenvio_memoria``: los PEs de memoria (sin aritmética) pueden reenviar
    datos que no cargaron ellos. ``horizonte``: ciclo máximo (por defecto
    ``4·n + 2·(filas + columnas) + 8``). ``holgura``: ciclos extra que se
    prueban en la frontera ocupada (por defecto ``2·(filas + columnas) + 8``).
    ``intentos_huecos``: cuántos ciclos libres anteriores a la frontera se
    prueban antes de saltar a ella (``None`` = ``holgura``); rellenar huecos
    viejos acorta el programa, pero cada intento fallido cuesta una búsqueda.
    ``ventana``: ciclos que cada búsqueda mira hacia atrás además del camino
    directo (por defecto ``filas + columnas``). ``max_candidatos``: corta la
    evaluación de PEs tras ese número de candidatos válidos (por defecto
    todos; cortar es mucho más rápido en mallas grandes, pero empeora el
    mapeo: matmul 6x10 pasa de 565 a 794 ciclos con 8).
    ``procesos``: procesos que prueban los candidatos de cada operación en
    paralelo (1 = secuencial). El resultado es idéntico al secuencial; solo
    cambia el tiempo. Requiere ``fork`` (Linux).
    """

    reenvio_memoria: bool = True
    horizonte: int = None
    holgura: int = None
    intentos_huecos: int = 8
    ventana: int = None
    max_candidatos: int = None
    procesos: int = 1


@dataclass
class ResultadoMapeo:
    """Lo que el bloque le entrega al skill 08 (y al agente Greedy)."""

    valido: bool
    motivo: str = ""
    ubicacion: dict = field(default_factory=dict)  # id de operación -> (PE, ciclo)
    programa: dict = field(default_factory=dict)  # PE -> [instrucción por ciclo]
    metricas: dict = field(default_factory=dict)

    @classmethod
    def invalido(cls, motivo, n_operaciones=0):
        return cls(False, motivo, metricas={"operaciones": n_operaciones, "ciclos": None})


class Copia:
    """Un valor en un registro de un PE, escrito en ``definicion`` y leído
    por última vez en ``ultimo``. ``abierta``: todavía le faltan consumidores,
    así que sigue vivo hasta nuevo aviso."""

    __slots__ = ("valor", "pe", "clase", "definicion", "ultimo", "abierta", "serie", "registro")

    def __init__(self, valor, pe, clase, definicion, abierta, serie):
        self.valor = valor
        self.pe = pe
        self.clase = clase  # 0 escalar, 1 vector
        self.definicion = definicion
        self.ultimo = definicion
        self.abierta = abierta
        self.serie = serie
        self.registro = None


_FALTA = object()


class Mapeador:
    def __init__(self, grafo, malla, compatibilidad, opciones=None):
        self.grafo = grafo
        self.malla = malla
        self.compat = compatibilidad
        self.opciones = opciones or Opciones()
        n_pes = len(malla)
        borde = malla.filas + malla.columnas
        self.holgura = self.opciones.holgura or 2 * borde + 8
        self.ventana = self.opciones.ventana if self.opciones.ventana is not None else borde
        self.max_candidatos = self.opciones.max_candidatos or len(malla)
        self.intentos_huecos = self.holgura if self.opciones.intentos_huecos is None else self.opciones.intentos_huecos
        self.horizonte = self.opciones.horizonte or 4 * len(grafo) + 2 * borde + 8
        self.n_pes = n_pes
        self.nombres = (malla.registros, malla.registros_vectoriales)
        self.capacidad = tuple(len(nombres) for nombres in self.nombres)
        self.clase = {op.id: int(es_vector(op.op)) for op in grafo.operaciones}
        self.transferencias = (compatibilidad.transferencia(False), compatibilidad.transferencia(True))
        self.slots = [dict() for _ in range(n_pes)]  # ciclo -> evento
        self.fin = [-1] * n_pes  # último ciclo ocupado de cada PE
        # Un banco de registros por PE y clase: índice pe + n_pes * clase.
        self.cerrados = [[] for _ in range(2 * n_pes)]  # registros ocupados por ciclo
        self.abiertos = [[] for _ in range(2 * n_pes)]  # ciclos de definición de copias abiertas
        self.copias = {}
        self.ubicacion = {}
        self.resultado = {}  # operación -> copia de su resultado
        self.lecturas = {}  # operación -> {valor: copia que lee}
        self.pendientes = {}
        self.diario = None  # deshacer de un intento
        self.series = 0
        self.motivo = ""
        self._distancia_minima = {}
        self._cuadrilla = None

    # -- cambios de estado que se pueden deshacer ---------------------------
    def _anotar(self, entrada):
        if self.diario is not None:
            self.diario.append(entrada)

    def _ocupar(self, pe, ciclo, evento):
        self.slots[pe][ciclo] = evento
        self._anotar(("slot", pe, ciclo, self.fin[pe]))
        self.fin[pe] = max(self.fin[pe], ciclo)

    def _sumar(self, banco, desde, hasta, delta):
        if hasta <= desde:
            return
        lista = self.cerrados[banco]
        if hasta > len(lista):
            lista.extend([0] * (hasta - len(lista)))
        for ciclo in range(desde, hasta):
            lista[ciclo] += delta
        self._anotar(("sumar", banco, desde, hasta, delta))

    def _abrir(self, banco, ciclo):
        self.abiertos[banco].append(ciclo)
        self._anotar(("abrir", banco, ciclo))

    def _cerrar_abierta(self, banco, ciclo):
        self.abiertos[banco].remove(ciclo)
        self._anotar(("cerrar", banco, ciclo))

    def _poner(self, diccionario, clave, valor):
        self._anotar(("dict", diccionario, clave, diccionario.get(clave, _FALTA)))
        diccionario[clave] = valor

    def _cambiar(self, copia, atributo, valor):
        self._anotar(("attr", copia, atributo, getattr(copia, atributo)))
        setattr(copia, atributo, valor)

    def _nueva_copia(self, valor, pe, definicion, abierta):
        self.series += 1
        copia = Copia(valor, pe, self.clase[valor], definicion, abierta, self.series)
        self.copias.setdefault(valor, []).append(copia)
        self._anotar(("copia", valor))
        if abierta:
            self._abrir(self._banco(copia), definicion)
        return copia

    def _deshacer(self):
        diario, self.diario = self.diario, None
        for entrada in reversed(diario):
            tipo = entrada[0]
            if tipo == "slot":
                del self.slots[entrada[1]][entrada[2]]
                self.fin[entrada[1]] = entrada[3]
            elif tipo == "sumar":
                _, banco, desde, hasta, delta = entrada
                lista = self.cerrados[banco]
                for ciclo in range(desde, hasta):
                    lista[ciclo] -= delta
            elif tipo == "abrir":
                self.abiertos[entrada[1]].remove(entrada[2])
            elif tipo == "cerrar":
                self.abiertos[entrada[1]].append(entrada[2])
            elif tipo == "dict":
                _, diccionario, clave, anterior = entrada
                if anterior is _FALTA:
                    del diccionario[clave]
                else:
                    diccionario[clave] = anterior
            elif tipo == "attr":
                setattr(entrada[1], entrada[2], entrada[3])
            elif tipo == "copia":
                lista = self.copias[entrada[1]]
                lista.pop()
                if not lista:
                    del self.copias[entrada[1]]

    # -- consultas -----------------------------------------------------------
    def _banco(self, copia):
        return copia.pe + self.n_pes * copia.clase

    def _registro_libre(self, pe, ciclo, clase):
        banco = pe + self.n_pes * clase
        lista = self.cerrados[banco]
        cantidad = lista[ciclo] if ciclo < len(lista) else 0
        for definicion in self.abiertos[banco]:
            if definicion <= ciclo:
                cantidad += 1
        return cantidad < self.capacidad[clase]

    def _cabe_abierta(self, pe, ciclo, clase):
        """¿Cabe un valor más de esa clase en ``pe`` desde ``ciclo`` hasta el final?"""
        banco = pe + self.n_pes * clase
        pico = max(self.cerrados[banco][ciclo:], default=0)
        return pico + len(self.abiertos[banco]) < self.capacidad[clase]

    def _primer_libre(self, pe, ciclo):
        while ciclo in self.slots[pe]:
            ciclo += 1
        return ciclo

    def _cota_memoria(self, op):
        return max((self.ubicacion[m][1] + 1 for m in op.memoria), default=0)

    def _distancia_a(self, instruccion, destino):
        clave = (instruccion, destino)
        if clave not in self._distancia_minima:
            self._distancia_minima[clave] = min(
                (self.malla.distancia(pe.indice, destino) for pe in self.malla.pes if instruccion in pe.instrucciones),
                default=None,
            )
        return self._distancia_minima[clave]

    # -- algoritmo -----------------------------------------------------------
    def mapear(self):
        if self.opciones.procesos > 1:
            return _mapear_en_cuadrilla(self, self.opciones.procesos)
        return self._mapear()

    def _mapear(self):
        n = len(self.grafo)
        try:
            self.grafo.validar()
            orden = self.grafo.orden_topologico()
        except ValueError as error:
            return ResultadoMapeo.invalido(f"grafo inválido: {error}", n)

        self.candidatos = {}
        for op in self.grafo.operaciones:
            if op.op in self.candidatos:
                continue
            candidatos = self.compat.candidatos(self.malla, op.op)
            if candidatos is None:
                return ResultadoMapeo.invalido(
                    f"la operación '{op.op}' (#{op.id}) no tiene entrada en la tabla de compatibilidad", n)
            if not candidatos:
                tipos = ", ".join(sorted({pe.letra for pe in self.malla.pes}))
                return ResultadoMapeo.invalido(
                    f"ningún PE de la malla ({tipos}) ejecuta {self.compat.instruccion(op.op)}, "
                    f"que necesita la operación '{op.op}' (#{op.id})", n)
            self.candidatos[op.op] = candidatos

        consumidores = self.grafo.consumidores()
        self.pendientes = {i: len(c) for i, c in consumidores.items()}
        for id_operacion in orden:
            op = self.grafo.operacion(id_operacion)
            if id_operacion in self.ubicacion:
                continue
            if op.op in PEREZOSAS and not op.entradas and consumidores[id_operacion]:
                continue  # se ubica con su primer consumidor
            if not self._ubicar(op):
                return ResultadoMapeo.invalido(self.motivo, n)
        return self._resultado()

    def _ubicar(self, op):
        for previa in op.memoria:
            if previa not in self.ubicacion and not self._ubicar_temprano(self.grafo.operacion(previa)):
                return False
        entradas = list(dict.fromkeys(op.entradas))
        if not entradas:
            return self._ubicar_temprano(op)

        cota_memoria = self._cota_memoria(op)
        cotas = []
        for destino in self.candidatos[op.op]:
            cota = cota_memoria
            for valor in entradas:
                if valor in self.ubicacion:
                    llegada = min(c.definicion + 1 + self.malla.distancia(c.pe, destino) for c in self.copias[valor])
                else:
                    fuente = self.grafo.operacion(valor)
                    distancia = self._distancia_a(self.compat.instruccion(fuente.op), destino)
                    llegada = self._cota_memoria(fuente) + 1 + distancia
                cota = max(cota, llegada)
            cotas.append((self._primer_libre(destino, cota), destino))
        cotas.sort()

        # Con cuadrilla, todos los candidatos se prueban desde el mismo estado, repartidos
        # entre procesos. _intentar prueba ciclos crecientes: si su resultado (con el
        # horizonte) cae después del límite de la selección, el límite lo habría descartado.
        pruebas = None
        if self._cuadrilla is not None:
            pruebas = self._cuadrilla.repartir(
                lambda i: self._intentar(op, cotas[i][1], cotas[i][0], entradas, self.horizonte, confirmar=False),
                len(cotas))

        mejor = None
        evaluados = 0
        for i, (inicio, destino) in enumerate(cotas):
            if mejor is not None and (inicio > mejor[0][0] or evaluados >= self.max_candidatos):
                break
            limite = mejor[0][0] if mejor is not None else self.horizonte
            if pruebas is None:
                clave = self._intentar(op, destino, inicio, entradas, limite, confirmar=False)
            else:
                clave = pruebas[i]
                if clave is not None and clave[0] > limite:
                    clave = None
            if clave is not None:
                evaluados += 1
                if mejor is None or clave < mejor[0]:
                    mejor = (clave, inicio)
        if mejor is None:
            self.motivo = (
                f"no se pudo rutear la operación #{op.id} ({op.texto()}): ningún PE compatible recibe "
                f"sus operandos antes del ciclo {self.horizonte} (registros o enlaces saturados)"
            )
            return False
        (_, _, destino), inicio = mejor
        self._intentar(op, destino, inicio, entradas, self.horizonte, confirmar=True)
        return True

    def _intentar(self, op, destino, inicio, entradas, limite, confirmar):
        """Prueba ``op`` en ``destino`` desde el ciclo ``inicio``; devuelve ``(ciclo, saltos, PE)``.

        Prueba ``intentos_huecos`` ciclos libres; si los huecos viejos de la
        línea de tiempo no sirven, salta a la frontera (después del último
        ciclo ocupado del PE y sus vecinos) y prueba ``holgura`` más. No pasa
        de ``limite``: un ciclo posterior al del mejor candidato ya no gana.
        """
        frontera = 1 + max(self.fin[pe] for pe in (destino, *(v for _, v in self.malla.vecinos[destino])))
        for valor in entradas:
            if valor not in self.ubicacion:
                # La fuente se carga en un PE compatible después de su último ciclo ocupado.
                candidatos = self.candidatos[self.grafo.operacion(valor).op]
                frontera = max(frontera, min(self.fin[pe] + 2 + self.malla.distancia(pe, destino)
                                             for pe in candidatos))
        ciclo, intentos, salto = inicio, 0, False
        while ciclo <= limite:
            if intentos >= (self.holgura if salto or ciclo >= frontera else self.intentos_huecos):
                if salto or ciclo >= frontera:
                    break
                ciclo, intentos, salto = frontera, 0, True
            if ciclo in self.slots[destino]:
                ciclo += 1
                continue
            if not self._hay_entradas(destino, ciclo, entradas):
                ciclo += 1
                intentos += 1
                continue
            self.diario = []
            serie = self.series
            clave = self._probar(op, destino, ciclo, entradas)
            if clave is not None and confirmar:
                self.diario = None
                return clave
            self._deshacer()
            self.series = serie  # el número de copia no sobrevive al deshacer
            if clave is not None:
                return clave
            ciclo += 1
            intentos += 1
        return None

    def _hay_entradas(self, destino, ciclo, entradas):
        """Condición necesaria (y barata) para ``_probar``: cada operando que no
        está ya en ``destino`` (ni se puede cargar ahí) necesita un ciclo libre
        de ``destino`` dentro de la ventana de búsqueda para recibirlo, y cada
        load/const sin ubicar, un PE compatible con un ciclo libre para
        ejecutarse y otro, después, para mandarlo."""
        faltan = 0
        clases = set()
        piso = max(0, ciclo - 2 - self.malla.filas - self.malla.columnas - self.ventana)
        for valor in entradas:
            if valor in self.ubicacion:
                if not any(c.pe == destino for c in self.copias[valor]):
                    faltan += 1
                    clases.add(self.clase[valor])
                continue
            fuente = self.grafo.operacion(valor)
            if self.compat.instruccion(fuente.op) in self.malla.pes[destino].instrucciones:
                continue
            faltan += 1
            clases.add(self.clase[valor])
            desde = max(piso, self._cota_memoria(fuente))
            if not any(self._dos_libres(pe, desde, ciclo - self.malla.distancia(pe, destino))
                       for pe in self.candidatos[fuente.op]):
                return False
        if not faltan:
            return True
        if any(self.transferencias[c][1] not in self.malla.pes[destino].instrucciones for c in clases):
            return False
        ocupado = self.slots[destino]
        enviar = {self.transferencias[c][0] for c in clases}
        vecinos = [v for _, v in self.malla.vecinos[destino] if enviar & self.malla.pes[v].instrucciones]
        for c in range(ciclo - 1, piso - 1, -1):
            if c not in ocupado and any(c not in self.slots[v] for v in vecinos):
                faltan -= 1
                if not faltan:
                    return True
        return False

    def _dos_libres(self, pe, desde, hasta):
        """¿Tiene ``pe`` dos ciclos libres en ``[desde, hasta]``?"""
        ocupado = self.slots[pe]
        libres = 0
        for c in range(desde, hasta + 1):
            if c not in ocupado:
                libres += 1
                if libres == 2:
                    return True
        return False

    def _probar(self, op, destino, ciclo, entradas):
        saltos = 0
        lecturas = {}
        for valor in entradas:
            encontrada = self._buscar(valor, destino, ciclo)
            if encontrada is None:
                return None
            copia, cantidad = self._confirmar_ruta(valor, *encontrada)
            lecturas[valor] = copia
            saltos += cantidad
        self._ocupar(destino, ciclo, ("op", op.id))
        if not self._guardar_resultado(op, destino, ciclo):
            return None
        self._poner(self.lecturas, op.id, lecturas)
        self._poner(self.ubicacion, op.id, (destino, ciclo))
        for valor in entradas:
            self._consumido(valor)
        return (ciclo, saltos, destino)

    def _guardar_resultado(self, op, pe, ciclo):
        if op.op in ("store", "vstore"):
            return True
        clase = self.clase[op.id]
        if self.pendientes[op.id] > 0:
            if not self._cabe_abierta(pe, ciclo, clase):
                return False
            copia = self._nueva_copia(op.id, pe, ciclo, abierta=True)
        else:
            # Nadie lo lee, pero escribe un registro en ese ciclo.
            if not self._registro_libre(pe, ciclo, clase):
                return False
            copia = self._nueva_copia(op.id, pe, ciclo, abierta=False)
            self._leer(copia, ciclo + 1)
        self._poner(self.resultado, op.id, copia)
        return True

    def _consumido(self, valor):
        self._poner(self.pendientes, valor, self.pendientes[valor] - 1)
        if self.pendientes[valor] == 0:
            for copia in self.copias[valor]:
                if copia.abierta:
                    self._cerrar_abierta(self._banco(copia), copia.definicion)
                    self._cambiar(copia, "abierta", False)
                    ultimo = max(copia.ultimo, copia.definicion + 1)
                    self._cambiar(copia, "ultimo", ultimo)
                    self._sumar(self._banco(copia), copia.definicion, ultimo, 1)

    def _leer(self, copia, ciclo):
        """La copia se lee en ``ciclo``: si está cerrada, su registro se alarga hasta ahí."""
        if ciclo <= copia.ultimo:
            return
        if not copia.abierta:
            self._sumar(self._banco(copia), max(copia.ultimo, copia.definicion), ciclo, 1)
        self._cambiar(copia, "ultimo", ciclo)

    def _ubicar_temprano(self, op):
        """Ubica en el primer ciclo posible una operación sin operandos."""
        cota = self._cota_memoria(op)
        mejor = None
        for pe in self.candidatos[op.op]:
            ciclo = self._primer_libre(pe, cota)
            clase = self.clase[op.id]
            while ciclo <= self.horizonte:
                abierta = self.pendientes[op.id] > 0
                if (abierta and self._cabe_abierta(pe, ciclo, clase)
                        or not abierta and self._registro_libre(pe, ciclo, clase)):
                    break
                ciclo = self._primer_libre(pe, ciclo + 1)
            if ciclo <= self.horizonte and (mejor is None or (ciclo, pe) < mejor):
                mejor = (ciclo, pe)
        if mejor is None:
            self.motivo = f"no hay registro ni ciclo libre para la operación #{op.id} ({op.texto()})"
            return False
        ciclo, pe = mejor
        self._ocupar(pe, ciclo, ("op", op.id))
        self._poner(self.ubicacion, op.id, (pe, ciclo))
        self._guardar_resultado(op, pe, ciclo)
        return True

    # -- Dijkstra en el tiempo -------------------------------------------------
    def _buscar(self, valor, destino, ciclo_ejecucion):
        """Ruta de ``valor`` hasta ``(destino, ciclo_ejecucion)``, buscando hacia atrás.

        Devuelve ``(fuente, nodos)``: ``fuente`` es una ``Copia`` existente o
        ``("nueva", pe, ciclo)`` para un load/const que todavía no está
        ubicado; ``nodos`` va de la fuente al destino.
        """
        perezosa = valor not in self.ubicacion
        clase = self.clase[valor]
        if perezosa:
            fuente_op = self.grafo.operacion(valor)
            instruccion = self.compat.instruccion(fuente_op.op)
            cota_fuente = self._cota_memoria(fuente_op)
            abierta = self.pendientes[valor] > 1
            piso = cota_fuente + 1
            directo = self._distancia_a(instruccion, destino)
        else:
            copias = self.copias[valor]
            piso = min(c.definicion + 1 for c in copias)
            directo = min(self.malla.distancia(c.pe, destino) for c in copias)
        # Hacia atrás solo se mira el camino directo más ``ventana`` ciclos.
        piso = max(piso, ciclo_ejecucion - 1 - directo - self.ventana)
        # Poda exacta: desde (p, t) una fuente está al menos a ``cerca(p)`` saltos,
        # así que si ``t - cerca(p) < piso`` ya no se llega a ninguna.
        if perezosa:
            cerca = [self._distancia_a(instruccion, pe) for pe in range(len(self.malla))]
        else:
            cerca = [min(self.malla.distancia(c.pe, pe) for c in copias) for pe in range(len(self.malla))]

        enviar, recibir = self.transferencias[clase]
        relevo_memoria = self.opciones.reenvio_memoria
        inicio = (destino, ciclo_ejecucion)
        saltos = {inicio: 0}
        previo = {}
        cola = [(-ciclo_ejecucion, 0, destino)]
        while cola:
            menos_ciclo, cantidad, pe = heapq.heappop(cola)
            ciclo = -menos_ciclo
            if saltos[(pe, ciclo)] != cantidad:
                continue
            # ¿Se puede tomar el valor de aquí?
            if perezosa:
                anterior = ciclo - 1
                if (anterior >= cota_fuente and anterior not in self.slots[pe]
                        and instruccion in self.malla.pes[pe].instrucciones
                        and (self._cabe_abierta(pe, anterior, clase) if abierta
                             else self._registro_libre(pe, anterior, clase))):
                    return ("nueva", pe, anterior), self._camino(previo, (pe, ciclo))
            else:
                for copia in copias:
                    if (copia.pe == pe and copia.definicion < ciclo
                            and (copia.abierta or ciclo <= copia.ultimo)):
                        return copia, self._camino(previo, (pe, ciclo))
            anterior = ciclo - 1
            if anterior < piso or not self._registro_libre(pe, anterior, clase):
                continue
            # Esperar en el mismo PE.
            nodo = (pe, anterior)
            if anterior - cerca[pe] >= piso and cantidad < saltos.get(nodo, cantidad + 1):
                saltos[nodo] = cantidad
                previo[nodo] = (pe, ciclo)
                heapq.heappush(cola, (-anterior, cantidad, pe))
            # Llegar desde un vecino: RECV aquí y SEND allá en el ciclo anterior.
            if anterior in self.slots[pe] or recibir not in self.malla.pes[pe].instrucciones:
                continue
            if pe != destino and not relevo_memoria and self.malla.es_memoria(pe):
                continue
            for _, vecino in self.malla.vecinos[pe]:
                if (anterior in self.slots[vecino] or enviar not in self.malla.pes[vecino].instrucciones
                        or anterior - cerca[vecino] < piso):
                    continue
                nodo = (vecino, anterior)
                if cantidad + 1 < saltos.get(nodo, cantidad + 2):
                    saltos[nodo] = cantidad + 1
                    previo[nodo] = (pe, ciclo)
                    heapq.heappush(cola, (-anterior, cantidad + 1, vecino))
        return None

    @staticmethod
    def _camino(previo, nodo):
        camino = [nodo]
        while nodo in previo:
            nodo = previo[nodo]
            camino.append(nodo)
        return camino

    def _confirmar_ruta(self, valor, fuente, nodos):
        """Reserva la ruta; devuelve la copia que queda en el destino y los saltos."""
        if isinstance(fuente, Copia):
            actual = fuente
        else:
            _, pe, ciclo = fuente
            self._ocupar(pe, ciclo, ("op", valor))
            self._poner(self.ubicacion, valor, (pe, ciclo))
            actual = self._nueva_copia(valor, pe, ciclo, abierta=self.pendientes[valor] > 1)
            self._poner(self.resultado, valor, actual)
        saltos = 0
        i = 0
        while True:
            j = i
            while j + 1 < len(nodos) and nodos[j + 1][0] == nodos[i][0]:
                j += 1
            pe, salida = nodos[j]
            self._leer(actual, salida)
            if j == len(nodos) - 1:
                return actual, saltos
            siguiente = nodos[j + 1][0]
            nueva = self._nueva_copia(valor, siguiente, salida, abierta=False)
            self._ocupar(pe, salida, ("send", self.malla.direccion[(pe, siguiente)], actual))
            self._ocupar(siguiente, salida, ("recv", self.malla.direccion[(siguiente, pe)], nueva))
            saltos += 1
            actual = nueva
            i = j + 1

    # -- salida ------------------------------------------------------------------
    def _asignar_registros(self):
        """Colorea los intervalos ``[definicion, ultimo)`` de cada PE y clase con sus registros."""
        por_banco = {}
        for copias in self.copias.values():
            for copia in copias:
                por_banco.setdefault((copia.pe, copia.clase), []).append(copia)
        for (pe, clase), copias in por_banco.items():
            copias.sort(key=lambda c: (c.definicion, c.ultimo, c.serie))
            libres = list(range(self.capacidad[clase]))
            ocupados = []  # (fin, registro)
            for copia in copias:
                while ocupados and ocupados[0][0] <= copia.definicion:
                    heapq.heappush(libres, heapq.heappop(ocupados)[1])
                if not libres:
                    pe_id = self.malla.pes[pe].id
                    raise RuntimeError(f"faltan registros en {pe_id}, ciclo {copia.definicion}")
                registro = heapq.heappop(libres)
                copia.registro = self.nombres[clase][registro]
                heapq.heappush(ocupados, (max(copia.ultimo, copia.definicion + 1), registro))

    def _texto(self, evento):
        tipo = evento[0]
        if tipo == "send":
            return f"{self.transferencias[evento[2].clase][0]} {evento[1]}, {evento[2].registro}"
        if tipo == "recv":
            return f"{self.transferencias[evento[2].clase][1]} {evento[1]}, {evento[2].registro}"
        op = self.grafo.operacion(evento[1])
        instruccion = self.compat.instruccion(op.op)
        if op.op in ("load", "vload"):
            return f"{instruccion} {self.resultado[op.id].registro}, {op.banco}[{op.indice}]"
        if op.op in ("store", "vstore"):
            return f"{instruccion} {op.banco}[{op.indice}], {self.lecturas[op.id][op.entradas[0]].registro}"
        if op.op == "const":
            return f"{instruccion} {self.resultado[op.id].registro}, {float(op.valor)!r}"
        fuentes = ", ".join(self.lecturas[op.id][e].registro for e in op.entradas)
        return f"{instruccion} {self.resultado[op.id].registro}, {fuentes}"

    def _resultado(self):
        try:
            self._asignar_registros()
        except RuntimeError as error:
            return ResultadoMapeo.invalido(f"error interno del mapeo: {error}", len(self.grafo))
        ciclos = max((max(s) + 1 for s in self.slots if s), default=0)
        programa = {}
        instrucciones = transferencias = 0
        for pe in self.malla.pes:
            eventos = self.slots[pe.indice]
            programa[pe.id] = [self._texto(eventos[c]) if c in eventos else "NOP" for c in range(ciclos)]
            instrucciones += len(eventos)
            transferencias += sum(1 for e in eventos.values() if e[0] == "send")
        slots = ciclos * len(self.malla)
        nops = slots - instrucciones
        metricas = {
            "operaciones": len(self.grafo),
            "ciclos": ciclos,
            "pes": len(self.malla),
            "slots": slots,
            "instrucciones": instrucciones,
            "transferencias": transferencias,
            "nops": nops,
            "ratio_nops": nops / slots if slots else 0.0,
        }
        ubicacion = {i: (self.malla.pes[pe].id, ciclo) for i, (pe, ciclo) in sorted(self.ubicacion.items())}
        return ResultadoMapeo(True, "", ubicacion, programa, metricas)


def mapear(grafo, malla, compatibilidad, opciones=None):
    """Mapea ``grafo`` en ``malla``. Nunca lanza por una configuración mala:
    devuelve un ``ResultadoMapeo`` inválido con el motivo."""
    return Mapeador(grafo, malla, compatibilidad, opciones).mapear()


class _Cuadrilla:
    """Procesos que ejecutan el mismo mapeo paso a paso (cada uno con su copia
    del estado). En cada ``_ubicar`` cada proceso prueba solo sus candidatos y
    se intercambian los resultados: como todos ven los mismos resultados y el
    estado es idéntico, todos toman las mismas decisiones."""

    def __init__(self, rango, n, canal=None, canales=()):
        self.rango = rango
        self.n = n
        self.canal = canal  # hijo: conexión con el padre
        self.canales = list(canales)  # padre: una conexión por hijo

    def repartir(self, probar, total):
        """``probar(i)`` para los candidatos ``i`` de este proceso; devuelve el
        resultado de todos (``{i: resultado}``)."""
        propios = {i: probar(i) for i in range(self.rango, total, self.n)}
        if self.canal is not None:
            self.canal.send(propios)
            return self._recibir(self.canal)
        resultados = dict(propios)
        for canal in self.canales:
            resultados.update(self._recibir(canal))
        for canal in self.canales:
            canal.send(resultados)
        return resultados

    @staticmethod
    def _recibir(canal):
        try:
            return canal.recv()
        except EOFError:
            raise RuntimeError("un proceso del mapeo terminó antes de tiempo") from None


def _mapear_en_cuadrilla(mapeador, n):
    contexto = multiprocessing.get_context("fork")
    pares = [contexto.Pipe() for _ in range(n - 1)]
    padres = [p for p, _ in pares]
    hijos = [h for _, h in pares]
    procesos = []
    try:
        for rango, canal in enumerate(hijos, start=1):
            otros = [h for h in hijos if h is not canal]
            proceso = contexto.Process(target=_correr_hijo,
                                       args=(mapeador, rango, n, canal, padres, otros))
            proceso.start()
            procesos.append(proceso)
        for canal in hijos:
            canal.close()
        mapeador._cuadrilla = _Cuadrilla(0, n, canales=padres)
        return mapeador._mapear()
    finally:
        mapeador._cuadrilla = None
        for canal in padres:
            canal.close()
        for proceso in procesos:
            proceso.join()


def _correr_hijo(mapeador, rango, n, canal, padres, otros):
    for conexion in (*padres, *otros):
        conexion.close()
    mapeador._cuadrilla = _Cuadrilla(rango, n, canal=canal)
    try:
        mapeador._mapear()
    except RuntimeError:
        # El padre ya terminó: su conexión se cerró y no hay más que hacer.
        pass
