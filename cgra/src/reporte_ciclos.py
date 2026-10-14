"""Análisis del programa ya cargado: cuántos ciclos se van en cómputo vs
comunicación, y el detalle de cada SEND entre PEs. No ejecuta la CGRA,
solo lee el ``Programa``."""

from pathlib import Path

from consola import Consola
from isa import ISA
from topologia import EN_ESPANOL

TITULO_REPORTE = "CICLOS: cómputo vs comunicación"


class ReporteCiclos:
    """Ciclos de cómputo, de comunicación e inactivos de un programa."""

    def __init__(self, programa, malla, nombre_programa=None, formas=None):
        """``formas``: ``{región: (filas, columnas)}`` de la memoria, para el encabezado."""
        self.programa = programa
        self.malla = malla
        self.nombre_programa = nombre_programa
        self.formas = formas

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
        """Cada SEND (o VSEND) del programa: ciclo, emisor, receptor y dirección."""
        envios = []
        for ciclo in self.programa.ciclos():
            for emisor, instruccion in self.programa.del_ciclo(ciclo).items():
                if instruccion["op"] in ISA.ENVIOS:
                    envios.append({
                        "ciclo": ciclo,
                        "emisor": emisor,
                        "receptor": self.malla.vecino(emisor, instruccion["dir"]),
                        "direccion": instruccion["dir"],
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
            "Pasos de comunicación entre PEs:",
        ]
        envios = self.comunicaciones()
        if not envios:
            lineas.append("  (este programa no manda datos entre PEs)")
        for envio in envios:
            lineas.append(
                f"  Ciclo {envio['ciclo']:>3}: {envio['emisor']} -> {envio['receptor']} "
                f"({EN_ESPANOL[envio['direccion']]})"
            )
        return lineas

    def mostrar(self):
        Consola.titulo(TITULO_REPORTE)
        Consola.caja(self.lineas())

    def guardar(self, ruta):
        """Guarda el reporte en un archivo de texto."""
        contenido = [TITULO_REPORTE, ""] + self.lineas()
        Path(ruta).write_text("\n".join(contenido) + "\n", encoding="utf-8")
