"""Análisis del programa ya cargado: cuántos ciclos se van en cómputo vs
comunicación, qué instrucciones ejecuta cada PE, qué PEs acceden a la
memoria (y a qué posiciones), cada SEND y todas las instrucciones por ciclo. No
ejecuta la CGRA, solo lee el ``Programa``."""

from collections import Counter, defaultdict
from pathlib import Path

from formato_numero import formatear_numero
from isa import ISA
from topologia import EN_ESPANOL

TITULO_REPORTE = "REPORTE DE SIMULACIÓN"

# Categorías de la tabla de instrucciones por PE.
CATEGORIAS = (
    ("memoria", ISA.MEMORIA),
    ("calculo", ISA.ARITMETICAS | ISA.ARITMETICAS_VECTORIALES | {"vredsum"}),
    ("comunicacion", ISA.COMUNICACION),
    ("otras", {"mov", "vbcast"}),
    ("nop", {"nop"}),
)


def _rangos(indices):
    """Agrupa posiciones consecutivas sin omitir ningún tramo."""
    ordenados = sorted(set(indices))
    tramos = []
    for indice in ordenados:
        if tramos and indice == tramos[-1][1] + 1:
            tramos[-1][1] = indice
        else:
            tramos.append([indice, indice])
    textos = [f"{a}" if a == b else f"{a}..{b}" for a, b in tramos]
    return ", ".join(textos)


class ReporteCiclos:
    """Ciclos de cómputo, de comunicación e inactivos de un programa."""

    def __init__(self, programa, malla, nombre_programa=None, formas=None, formato=None,
                 inicial=None, final=None, traza=None):
        """``formas``: ``{región: (filas, columnas)}`` de la memoria, para el encabezado.
        ``formato``: el ``FormatoCGRA``, para el tipo de cada PE y el ancho de los vectores."""
        self.programa = programa
        self.malla = malla
        self.nombre_programa = nombre_programa
        self.formas = formas
        self.formato = formato
        self.inicial = inicial
        self.final = final
        self.efectos = defaultdict(list)
        if traza is not None:
            for entrada in traza.entradas:
                self.efectos[entrada["ciclo"]].append(entrada)

    @staticmethod
    def _seccion(numero, titulo):
        return ["", f"{numero}. {titulo}", "-" * 80]

    @staticmethod
    def _tabla(cabecera, filas):
        anchos = [max(len(str(f[i])) for f in [cabecera] + filas) for i in range(len(cabecera))]
        def linea(fila):
            return "  " + " | ".join(str(celda).ljust(ancho) for celda, ancho in zip(fila, anchos))
        return [linea(cabecera), "  " + "-+-".join("-" * ancho for ancho in anchos)] + [linea(fila) for fila in filas]

    def _escrituras(self):
        bancos = defaultdict(Counter)
        for instrucciones in self.programa.instrucciones.values():
            for inst in instrucciones.values():
                if inst["op"] in {"st", "vst"}:
                    bancos[inst["bank"]].update(range(inst["index"], inst["index"] + self._palabras(inst["op"])))
        return bancos

    @staticmethod
    def _clase_ciclo(instrucciones):
        operaciones = [i["op"] for i in instrucciones.values()]
        if any(op in ISA.COMUNICACION for op in operaciones):
            return "comunicación"
        return "inactivo" if all(op == "nop" for op in operaciones) else "cómputo"

    def _tipo(self, pe_id):
        return self.formato.nombre_tipo(pe_id) if self.formato else "-"

    def _palabras(self, op):
        """Palabras que lee o escribe una instrucción de memoria."""
        if op in {"vld", "vst"}:
            return (self.formato.ancho_vector if self.formato else None) or 1
        return 1

    def por_pe(self):
        """``{pe_id: Counter(op)}``: cuántas veces ejecuta cada instrucción cada PE."""
        return {pe_id: Counter(instruccion["op"] for instruccion in instrucciones.values())
                for pe_id, instrucciones in self.programa.instrucciones.items()}

    def accesos_memoria(self):
        """``{pe_id: {"lee"/"escribe": {región: [índices]}}}``, solo de los PEs que acceden a memoria."""
        accesos = {}
        for pe_id, instrucciones in self.programa.instrucciones.items():
            for instruccion in instrucciones.values():
                op = instruccion["op"]
                if op not in ISA.MEMORIA:
                    continue
                sentido = "lee" if op in {"ld", "vld"} else "escribe"
                indices = range(instruccion["index"], instruccion["index"] + self._palabras(op))
                accesos.setdefault(pe_id, {}).setdefault(sentido, {}).setdefault(instruccion["bank"], []).extend(indices)
        return accesos

    def contar(self):
        """Clasifica cada ciclo según lo que hacen los PEs en él.

        Un ciclo es "de comunicación" si algún PE ejecuta SEND o RECV,
        "de cómputo" si ningún PE comunica pero al menos uno hace algo
        (ADD, LD, MOV, etc.), e "inactivo" si todos están en NOP.
        """
        comunicacion = computo = inactivo = 0
        ciclos = self.programa.ciclos()
        for ciclo in ciclos:
            operaciones = [instruccion["op"] for instruccion in self.programa.del_ciclo(ciclo).values()]
            if any(op in ISA.COMUNICACION for op in operaciones):
                comunicacion += 1
            elif all(op == "nop" for op in operaciones):
                inactivo += 1
            else:
                computo += 1
        return {"total": len(ciclos), "comunicacion": comunicacion, "computo": computo, "inactivo": inactivo}

    def comunicaciones(self):
        """Cada SEND (o VSEND) del programa: ciclo, emisor, receptor, dirección e instrucción."""
        envios = []
        for ciclo in self.programa.ciclos():
            for emisor, instruccion in self.programa.del_ciclo(ciclo).items():
                if instruccion["op"] in ISA.ENVIOS:
                    envios.append({
                        "ciclo": ciclo,
                        "emisor": emisor,
                        "receptor": self.malla.vecino(emisor, instruccion["dir"]),
                        "direccion": instruccion["dir"],
                        "instruccion": ISA.texto(instruccion),
                    })
        return envios

    def lineas(self):
        """Las líneas de texto del reporte."""
        estadisticas = self.contar()
        total = estadisticas["total"]

        def con_porcentaje(valor):
            return f"{valor} ({100 * valor / total:.1f}%)" if total else str(valor)

        filas, columnas = self.malla.filas, self.malla.columnas
        nombre = Path(self.nombre_programa).name if self.nombre_programa else "programa cargado"
        lineas = ["=" * 80, f"{TITULO_REPORTE}: {nombre.upper()}", "=" * 80]
        lineas += self._seccion(1, "CONFIGURACIÓN")
        lineas.append(f"Malla de la CGRA:       {filas}x{columnas} ({filas * columnas} PEs)")
        if self.nombre_programa:
            lineas.append(f"Programa:               {self.nombre_programa}")
        if self.formas:
            lineas.append("Memoria:                " + ", ".join(
                f"{nombre} ({f}x{c})" for nombre, (f, c) in self.formas.items()
            ))
        if self.formato:
            lineas += [f"Ancho vectorial:        {self.formato.ancho_vector or 'sin vectores'}",
                       f"Capacidad de memoria:   {self.formato.palabras_memoria} palabras",
                       "", "Mapa de tipos:"]
            for inicio in range(0, columnas, 8):
                columnas_bloque = range(inicio, min(inicio + 8, columnas))
                lineas += self._tabla(
                    ["Fila"] + [f"Columna {c}" for c in columnas_bloque],
                    [[r] + [self.formato.malla[r][c] for c in columnas_bloque] for r in range(filas)])
            lineas += [""] + self.formato.leyenda()
        lineas += self._seccion(2, "RESULTADO")
        if self.final is None:
            lineas.append("Resultados no disponibles: este reporte solo recibió las instrucciones.")
        else:
            escritas = self._escrituras()
            if not escritas:
                lineas.append("El programa no escribió regiones de memoria.")
            for banco in sorted(escritas, key=lambda b: len(self.final[b])):
                indices = escritas[banco]
                valores = self.final[banco]
                if len(valores) == 1:
                    lineas.append(f"{banco} = {formatear_numero(valores[0])}")
                else:
                    lineas.append(f"{banco}: {len(valores)} valores (ver memoria final).")
                lineas.append(f"  Posiciones escritas: {len(indices)} de {len(valores)}; "
                              f"posiciones con escrituras repetidas: {sum(n > 1 for n in indices.values())}.")
            lineas.append("Comparación con referencia: no realizada por el simulador.")
        lineas += self._seccion(3, "RESUMEN DE LA EJECUCIÓN")
        lineas += [
            f"Total de ciclos:        {total}",
            f"Ciclos de cómputo:      {con_porcentaje(estadisticas['computo'])}",
            f"Ciclos de comunicación: {con_porcentaje(estadisticas['comunicacion'])}",
            f"Ciclos inactivos:       {con_porcentaje(estadisticas['inactivo'])}",
            "",
            "Criterio: comunicación si algún PE envía o recibe; cómputo si no hay",
            "comunicaciones y algún PE trabaja; inactivo si todos ejecutan NOP.",
            "Un ciclo de comunicación también puede contener cálculos en otros PEs.",
        ]
        lineas += self._seccion(4, "DISTRIBUCIÓN DEL TRABAJO")
        lineas.append("Funciones observadas en las instrucciones (incluyen acumulación y reducción):")
        conteos = self.por_pe()
        for pe_id in self.malla.ids():
            conteo = conteos[pe_id]
            escalar = sum(conteo[op] for op in ISA.ARITMETICAS)
            vectorial = sum(conteo[op] for op in ISA.ARITMETICAS_VECTORIALES | {"vredsum", "vbcast"})
            memoria = sum(conteo[op] for op in ISA.MEMORIA)
            tareas = []
            if escalar:
                tareas.append("cálculo y acumulación escalar")
            if vectorial:
                ancho = self.formato.ancho_vector if self.formato else self.malla.pe(pe_id).ancho_vector
                tareas.append(f"cálculo vectorial con bloques de {ancho} elementos")
            if memoria:
                if conteo["ld"] or conteo["vld"]:
                    tareas.append("carga operandos de memoria")
                if conteo["st"] or conteo["vst"]:
                    tareas.append("almacena resultados")
            if not tareas:
                tareas.append("comunica datos" if any(conteo[op] for op in ISA.COMUNICACION) else "sin cálculo ni acceso a memoria")
            lineas.append(f"  {pe_id} ({self._tipo(pe_id)}): " + "; ".join(tareas) + ".")
        lineas += [""] + self._lineas_memoria()
        lineas += self._seccion(5, "ACTIVIDAD POR PE")
        lineas += self._lineas_por_pe(total)
        lineas += ["", "Cada fila suma el total de ciclos: una instrucción por PE por ciclo.",
                   "Ocupación = porcentaje de ciclos en los que el PE no ejecuta NOP."]
        lineas += self._seccion(6, "EJECUCIÓN CICLO POR CICLO")
        lineas += self._lineas_ciclos()
        lineas += self._seccion(7, "MEMORIA FINAL")
        lineas += self._lineas_memoria_final()
        lineas += ["", "=" * 80, "FIN DEL REPORTE", "=" * 80]
        return lineas

    def _lineas_ciclos(self):
        """Incluye todos los ciclos y PEs, también los que solo ejecutan NOP."""
        lineas = ["Detalle de todos los ciclos (una instrucción por PE, incluidos NOP):"]
        envios = defaultdict(list)
        for envio in self.comunicaciones():
            envios[envio["ciclo"]].append(envio)
        if not envios:
            lineas.append("  (este programa no manda datos entre PEs)")
        for ciclo in self.programa.ciclos():
            lineas += ["", f"  Ciclo {ciclo}:"]
            instrucciones = self.programa.del_ciclo(ciclo)
            lineas.append(f"  Tipo: {self._clase_ciclo(instrucciones)}")
            for inicio in range(0, self.malla.columnas, 4):
                columnas = range(inicio, min(inicio + 4, self.malla.columnas))
                filas = []
                for r in range(self.malla.filas):
                    fila = [r]
                    for c in columnas:
                        pe_id = self.malla.id_de(r, c)
                        fila.append(f"{pe_id} ({self._tipo(pe_id)}): {ISA.texto(instrucciones[pe_id])}")
                    filas.append(fila)
                lineas += self._tabla(["Fila"] + [f"Columna {c}" for c in columnas], filas)
            for envio in envios[ciclo]:
                lineas.append(f"    Ciclo {ciclo:>3}: {envio['emisor']} -> {envio['receptor']} "
                              f"({EN_ESPANOL[envio['direccion']]})  {envio['instruccion']}")
            if self.efectos[ciclo]:
                lineas.append("  Valores observados (traza):")
                for entrada in self.efectos[ciclo]:
                    lineas.append(f"    {entrada['pe']}: {entrada['efecto']}")
        return lineas

    def _lineas_memoria_final(self):
        if self.final is None:
            return ["Memoria final no disponible: no se proporcionó una ejecución."]
        lineas = []
        for banco in self._escrituras():
            valores = self.final[banco]
            filas, columnas = self.formas[banco]
            lineas += [f"Región {banco} ({filas}x{columnas}):"]
            if len(valores) == 1:
                lineas.append(f"  {banco}[0] = {formatear_numero(valores[0])}")
            elif filas == 1 or columnas == 1:
                for inicio in range(0, len(valores), 10):
                    grupo = valores[inicio:inicio + 10]
                    lineas.append(f"  [{inicio}..{inicio + len(grupo) - 1}] "
                                  + " ".join(formatear_numero(v) for v in grupo))
            else:
                for r in range(filas):
                    lineas.append(f"  Fila {r}: [" + ", ".join(
                        formatear_numero(v) for v in valores[r * columnas:(r + 1) * columnas]) + "]")
            lineas.append("")
        return lineas or ["El programa no escribió regiones de memoria."]

    def _lineas_por_pe(self, total):
        """Tabla con las instrucciones de cada PE por categoría, y el detalle de cada una."""
        conteos = self.por_pe()
        ancho_pe = max(len(pe_id) for pe_id in conteos)
        ancho_tipo = max([len(self._tipo(pe_id)) for pe_id in conteos] + [4])
        lineas = [
            "Instrucciones por PE (memoria = LD/ST/VLD/VST, cálculo = ADD..DIV, V.. y VREDSUM,",
            "comunicación = SEND/RECV/VSEND/VRECV, otras = MOV/VBCAST):",
            f"  {'PE':<{ancho_pe}}  {'Tipo':<{ancho_tipo}}  Memoria  Cálculo  Comunic.  Otras    NOP  Ocupado",
        ]
        for pe_id in self.malla.ids():
            conteo = conteos[pe_id]
            por_categoria = [sum(conteo[op] for op in ops) for _, ops in CATEGORIAS]
            ocupado = 100 * (total - conteo["nop"]) / total if total else 0
            lineas.append(
                f"  {pe_id:<{ancho_pe}}  {self._tipo(pe_id):<{ancho_tipo}}  "
                + "  ".join(f"{valor:>{ancho}}" for valor, ancho in zip(por_categoria, (7, 7, 8, 5, 5)))
                + f"  {ocupado:>6.1f}%"
            )
        lineas += ["", "Instrucciones que ejecuta cada PE (sin NOP):"]
        orden = {op.lower(): i for i, op in enumerate(ISA.INSTRUCCIONES)}
        for pe_id in self.malla.ids():
            usadas = sorted((op for op in conteos[pe_id] if op != "nop"), key=lambda op: orden.get(op, len(orden)))
            detalle = ", ".join(f"{op.upper()} x{conteos[pe_id][op]}" for op in usadas) or "solo NOP"
            lineas.append(f"  {pe_id:<{ancho_pe}} ({self._tipo(pe_id)}): {detalle}")
        return lineas

    def _lineas_memoria(self):
        """Qué PE lee o escribe qué región de la memoria, y en qué posiciones."""
        accesos = self.accesos_memoria()
        lineas = ["Accesos a memoria (posiciones dentro de cada región, y cuántas palabras lee o escribe en total):"]
        if not accesos:
            return lineas + ["  (ningún PE accede a la memoria)"]
        for pe_id in self.malla.ids():
            if pe_id not in accesos:
                continue
            partes = []
            for sentido, nombre in (("lee", "lectura"), ("escribe", "escritura")):
                for banco, indices in accesos[pe_id].get(sentido, {}).items():
                    veces = f"{len(indices)} {nombre}" + ("s" if len(indices) != 1 else "")
                    partes.append(f"{sentido} {banco}[{_rangos(indices)}] ({veces})")
            lineas.append(f"  {pe_id} ({self._tipo(pe_id)}): " + "; ".join(partes))
        sin_memoria = [pe_id for pe_id in self.malla.ids() if pe_id not in accesos]
        if sin_memoria:
            lineas.append(f"  Sin acceso a memoria: {len(sin_memoria)} PEs")
        return lineas

    def mostrar(self):
        print("\n".join(self.lineas()))

    def guardar(self, ruta):
        """Guarda el reporte en un archivo de texto."""
        Path(ruta).write_text("\n".join(self.lineas()) + "\n", encoding="utf-8")
