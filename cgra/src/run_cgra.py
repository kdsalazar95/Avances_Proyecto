"""Punto de entrada para ejecutar un programa en una CGRA de tamaño NxM.

La CGRA funciona como un procesador: no se le dice qué operación va a
correr. Recibe la carpeta de un programa ya compilado por los skills y
ejecuta lo que encuentra ahí:

    programas/<operacion>/
    ├── pe_instructions/PE{fila}{columna}.txt   un programa por PE (skill 08)
    └── memoria.bin                             memoria inicial, la genera operaciones.c (skill 02)

Basta con el nombre de la operación (``matmul``) para usar
programas/<operacion>/, y ``todas`` corre, una tras otra, todas las
operaciones de la tabla OPERACIONES de operaciones/operaciones.c. La CGRA en sí (tamaño de la malla y de la
memoria) se describe en src/cgra.json, que también leen los skills.
"""

import argparse
import re
from pathlib import Path

from cargador import Cargador
from cgra import CGRA
from consola import Consola
from formato_cgra import RUTA_POR_DEFECTO, FormatoCGRA
from memoria import ImagenMemoria
from reporte_ciclos import ReporteCiclos
from resultados import Resultados
from traza import Traza

# Carpeta donde los skills dejan cada operación compilada, y la fuente C
# cuya tabla OPERACIONES dice qué operaciones existen.
RAIZ_CGRA = Path(__file__).resolve().parent.parent
CARPETA_PROGRAMAS = RAIZ_CGRA / "programas"
FUENTE_OPERACIONES = RAIZ_CGRA / "operaciones" / "operaciones.c"


class SimuladorCGRA:
    """Enciende la CGRA, le carga un programa, lo ejecuta y muestra los resultados."""

    def __init__(self, ruta_formato=RUTA_POR_DEFECTO, carpeta_programas=CARPETA_PROGRAMAS):
        self.ruta_formato = ruta_formato
        self.carpeta_programas = Path(carpeta_programas)

    def carpeta_del_programa(self, nombre):
        """Una ruta existente se usa tal cual; si no, ``nombre`` es una operación
        de programas/ (``matmul`` -> programas/matmul)."""
        ruta = Path(nombre)
        if not ruta.is_dir() and (self.carpeta_programas / nombre).is_dir():
            return self.carpeta_programas / nombre
        return ruta

    @staticmethod
    def operaciones(fuente=FUENTE_OPERACIONES):
        """Los nombres de la tabla OPERACIONES de operaciones.c, en su orden."""
        texto = Path(fuente).read_text(encoding="utf-8")
        tabla = re.search(r"OPERACIONES\[\]\s*=\s*\{(.*?)\};", texto, re.S)
        if not tabla:
            raise SystemExit(f"{fuente}: no se encontró la tabla OPERACIONES")
        return re.findall(r'\{\s*"(\w+)"\s*,', tabla.group(1))

    def correr_todas(self, mostrar_traza=False, mostrar_ciclos=False, write_back=False, fuente=FUENTE_OPERACIONES):
        """Corre todas las operaciones, una tras otra, cada una con su memoria y
        sus programas en programas/<operacion>/.

        Si una falla, se anota el error y se sigue con la siguiente. Al final
        se muestra una línea por operación. Devuelve ``{operacion: ciclos o error}``.
        """
        resultados = {}
        for operacion in self.operaciones(fuente):
            Consola.titulo(f"OPERACIÓN: {operacion}")
            try:
                ciclos = self.correr(operacion, mostrar_traza=mostrar_traza,
                                     mostrar_ciclos=mostrar_ciclos, write_back=write_back)
                resultados[operacion] = ciclos
            except SystemExit as error:
                print(f"  {error}")
                resultados[operacion] = str(error)

        Consola.titulo("RESUMEN")
        Consola.caja([
            f"{operacion:<12} ok     {valor} ciclos" if isinstance(valor, int) else f"{operacion:<12} error  {valor}"
            for operacion, valor in resultados.items()
        ])
        return resultados

    def correr(self, programa, ruta_memoria=None, ruta_instrucciones=None,
               mostrar_traza=False, mostrar_ciclos=False, ruta_reporte=None, write_back=False):
        """Corre un programa y devuelve cuántos ciclos duró."""
        carpeta = self.carpeta_del_programa(programa)
        ruta_memoria = Path(ruta_memoria) if ruta_memoria else carpeta / "memoria.bin"
        ruta_instrucciones = Path(ruta_instrucciones) if ruta_instrucciones else carpeta / "pe_instructions"
        if not ruta_memoria.is_file():
            raise SystemExit(
                f"No existe {ruta_memoria}. Se genera compilando y ejecutando operaciones/operaciones.c (skill 02)."
            )

        # 1. Encender: la CGRA (según cgra.json) arranca con la memoria en cero.
        try:
            cgra = CGRA(FormatoCGRA.cargar(self.ruta_formato))
        except ValueError as error:
            raise SystemExit(f"Formato de la CGRA inválido: {error}")
        # 2. Cargar: se copia la imagen de memoria y se prepara el programa.
        try:
            programa_cargado = Cargador(cgra).cargar(ruta_memoria, ruta_instrucciones)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(f"No se pudo cargar el programa: {error}")
        inicial, formas = cgra.memoria.regiones()

        Consola.titulo(f"CGRA {cgra.filas}x{cgra.columnas}: ejecutando {len(programa_cargado.ciclos())} ciclos")
        print(f"  Formato:  {self.ruta_formato}")
        for linea in cgra.formato.mapa() + cgra.formato.leyenda():
            print(f"            {linea}")
        print(f"  Programa: {carpeta}")
        print(f"  Memoria:  {ruta_memoria} ({len(formas)} regiones) cargada en una memoria de {len(cgra.memoria)} palabras")

        # 3. Ejecutar.
        traza = Traza() if mostrar_traza else None
        cgra.ejecutar(programa_cargado, traza)
        if traza is not None:
            traza.mostrar()
        final, _ = cgra.memoria.regiones()
        Resultados(inicial, final, formas).mostrar()

        # 4. Reportar. El reporte de ciclos se guarda en todas las corridas,
        #    en la carpeta del programa, y se reescribe en la siguiente.
        reporte = ReporteCiclos(programa_cargado, cgra.malla, str(carpeta), formas)
        if mostrar_ciclos:
            reporte.mostrar()
        ruta_reporte = Path(ruta_reporte) if ruta_reporte else carpeta / "reporte_ciclos.txt"
        ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
        reporte.guardar(ruta_reporte)
        print(f"\n  Reporte de ciclos guardado en {ruta_reporte}")
        if write_back:
            ImagenMemoria.desde_regiones(final, formas).guardar(ruta_memoria)
            print(f"  Memoria con los resultados guardada en {ruta_memoria}")
        return len(programa_cargado.ciclos())


def main():
    parser = argparse.ArgumentParser(description="Ejecuta un programa en una CGRA de tamaño NxM.")
    parser.add_argument(
        "programa",
        help="Operación de programas/ (ej. matmul), 'todas', o carpeta con pe_instructions/ y memoria.bin",
    )
    parser.add_argument(
        "--instrucciones", metavar="DIR", help="Usa estos PE*.txt en vez de <programa>/pe_instructions",
    )
    parser.add_argument("--memoria", metavar="RUTA", help="Usa esta memoria en vez de <programa>/memoria.bin")
    parser.add_argument(
        "--cgra", metavar="JSON", default=RUTA_POR_DEFECTO,
        help="Formato de la CGRA: tamaño de la malla y de la memoria (por defecto src/cgra.json)",
    )
    parser.add_argument(
        "--write-back", action="store_true",
        help="Guarda la memoria resultante (con los resultados) en el mismo memoria.bin",
    )
    parser.add_argument(
        "--traza", action="store_true",
        help="Muestra, ciclo por ciclo, qué instrucción ejecutó cada PE y qué escribió",
    )
    parser.add_argument("--ciclos", action="store_true", help="Muestra cuántos ciclos se van en cómputo vs comunicación")
    parser.add_argument(
        "--reporte-ciclos", metavar="RUTA",
        help="Guarda el reporte de ciclos en otra ruta (por defecto <programa>/reporte_ciclos.txt)",
    )
    argumentos = parser.parse_args()
    simulador = SimuladorCGRA(argumentos.cgra)
    if argumentos.programa == "todas":
        # Cada operación usa su propia memoria, programas y reporte.
        por_operacion = [opcion for opcion, valor in (
            ("--memoria", argumentos.memoria),
            ("--instrucciones", argumentos.instrucciones),
            ("--reporte-ciclos", argumentos.reporte_ciclos),
        ) if valor]
        if por_operacion:
            parser.error(f"{', '.join(por_operacion)} no se puede usar con 'todas'")
        simulador.correr_todas(
            mostrar_traza=argumentos.traza,
            mostrar_ciclos=argumentos.ciclos,
            write_back=argumentos.write_back,
        )
        return
    simulador.correr(
        argumentos.programa,
        ruta_memoria=argumentos.memoria,
        ruta_instrucciones=argumentos.instrucciones,
        mostrar_traza=argumentos.traza,
        mostrar_ciclos=argumentos.ciclos,
        ruta_reporte=argumentos.reporte_ciclos,
        write_back=argumentos.write_back,
    )


if __name__ == "__main__":
    main()
