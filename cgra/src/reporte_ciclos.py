"""Análisis del programa ya cargado: cuántos ciclos se van en cómputo vs
comunicación, qué instrucciones ejecuta cada PE, qué PEs acceden a la
memoria (y a qué posiciones) y el detalle de cada SEND entre PEs. No
ejecuta la CGRA, solo lee el ``Programa``."""

from collections import Counter
from pathlib import Path

from consola import Consola
from isa import ISA
from topologia import EN_ESPANOL

TITULO_REPORTE = "CICLOS: cómputo vs comunicación"

# Categorías de la tabla de instrucciones por PE.
CATEGORIAS = (
    ("memoria", ISA.MEMORIA),
    ("calculo", ISA.ARITMETICAS | ISA.ARITMETICAS_VECTORIALES | {"vredsum"}),
    ("comunicacion", ISA.COMUNICACION),
    ("otras", {"mov", "vbcast"}),
    ("nop", {"nop"}),
)


def _rangos(indices):
    """``[0, 1, 2, 3, 8, 9]`` -> ``"0..3, 8..9"``; con muchos tramos, solo el primero y el último."""
    ordenados = sorted(set(indices))
    tramos = []
    for indice in ordenados:
        if tramos and indice == tramos[-1][1] + 1:
            tramos[-1][1] = indice
        else:
            tramos.append([indice, indice])
    textos = [f"{a}" if a == b else f"{a}..{b}" for a, b in tramos]
    if len(textos) > 6:
        textos = textos[:3] + ["..."] + textos[-2:]
    return ", ".join(textos)


class ReporteCiclos:
    """Ciclos de cómputo, de comunicación e inactivos de un programa."""

    def __init__(self, programa, malla, nombre_programa=None, formas=None, formato=None):
        """``formas``: ``{región: (filas, columnas)}`` de la memoria, para el encabezado.
        ``formato``: el ``FormatoCGRA``, para el tipo de cada PE y el ancho de los vectores."""
        self.programa = programa
        self.malla = malla
        self.nombre_programa = nombre_programa
        self.formas = formas
        self.formato = formato

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
        lineas = [f"Malla de la CGRA:       {filas}x{columnas} ({filas * columnas} PEs)"]
        if self.nombre_programa:
            lineas.append(f"Programa:               {self.nombre_programa}")
        if self.formas:
            lineas.append("Memoria:                " + ", ".join(
                f"{nombre} ({f}x{c})" for nombre, (f, c) in self.formas.items()
            ))
        if len(lineas) > 1:
            lineas.append("")

        lineas += [
            f"Total de ciclos:        {total}",
            f"Ciclos de cómputo:      {con_porcentaje(estadisticas['computo'])}",
            f"Ciclos de comunicación: {con_porcentaje(estadisticas['comunicacion'])}",
            f"Ciclos inactivos:       {con_porcentaje(estadisticas['inactivo'])}",
            "",
        ]
        lineas += self._lineas_por_pe(total)
        lineas += [""] + self._lineas_memoria()
        lineas += ["", "Pasos de comunicación entre PEs:"]
        envios = self.comunicaciones()
        if not envios:
            lineas.append("  (este programa no manda datos entre PEs)")
        for envio in envios:
            lineas.append(
                f"  Ciclo {envio['ciclo']:>3}: {envio['emisor']} -> {envio['receptor']} "
                f"({EN_ESPANOL[envio['direccion']]})  {envio['instruccion']}"
            )
        return lineas

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
        Consola.titulo(TITULO_REPORTE)
        Consola.caja(self.lineas())

    def guardar(self, ruta):
        """Guarda el reporte en un archivo de texto."""
        contenido = [TITULO_REPORTE, ""] + self.lineas()
        Path(ruta).write_text("\n".join(contenido) + "\n", encoding="utf-8")
