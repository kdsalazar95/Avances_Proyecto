"""Punto de entrada para ejecutar un programa en una CGRA de tamaño NxM.

La CGRA funciona como un procesador: no se le dice qué operación va a
correr. Recibe la carpeta de un programa ya compilado por los skills y
ejecuta lo que encuentra ahí:

    src/<programa>/
    ├── pe_instructions/PE{fila}{columna}.txt   un programa por PE (skill 08)
    └── memoria.bin                             memoria inicial, la genera el .c (skill 02)

La CGRA en sí (tamaño de la malla, de la memoria y tipo de cada PE) se
describe en src/cgra.json, que también leen los skills.
"""

import argparse
from pathlib import Path

from cargador import cargar_programa
from cgra import CGRA
from formato_cgra import RUTA_POR_DEFECTO, cargar_formato
from memoria_binaria import escribir_memoria
from programas_pe import ejecutar_cgra
from reportes_ciclos import guardar_reporte_ciclos, mostrar_estadisticas_ciclos
from resultados import mostrar_resultados
from formato_texto import imprimir_titulo


def ejecutar_y_mostrar(argumentos):
    """Enciende la CGRA, le carga el programa (memoria + PEs), la ejecuta y muestra los resultados."""
    programa = Path(argumentos.programa)
    ruta_memoria = Path(argumentos.memoria) if argumentos.memoria else programa / "memoria.bin"
    ruta_instrucciones = Path(argumentos.instrucciones) if argumentos.instrucciones else programa / "pe_instructions"
    if not ruta_memoria.is_file():
        raise SystemExit(
            f"No existe {ruta_memoria}. Se genera compilando y ejecutando el .c del programa (skill 02)."
        )

    # 1. Encender: la CGRA (según su formato, cgra.json) arranca con la
    #    memoria en cero y sin programas.
    try:
        formato = cargar_formato(argumentos.cgra)
    except ValueError as error:
        raise SystemExit(f"Formato de la CGRA inválido: {error}")
    cgra = CGRA(formato)
    # 2. Cargar: se copia la imagen de memoria y cada PE recibe su programa.
    try:
        programas = cargar_programa(cgra, ruta_memoria, ruta_instrucciones)
    except ValueError as error:
        raise SystemExit(f"No se pudo cargar el programa: {error}")
    inicial, formas = cgra.regiones()
    ciclos = len(next(iter(programas.values())))

    imprimir_titulo(f"CGRA {cgra.filas}x{cgra.columnas}: ejecutando {ciclos} ciclos")
    print(f"  Formato:  {argumentos.cgra}")
    for linea in formato.mapa():
        print(f"            {linea}")
    for linea in formato.leyenda():
        print(f"            {linea}")
    print(f"  Programa: {programa}")
    print(f"  Memoria:  {ruta_memoria} ({len(formas)} regiones) cargada en una memoria de {len(cgra.memoria)} palabras")
    # 3. Ejecutar.
    ejecutar_cgra(cgra.memoria, programas, cgra.filas, cgra.columnas,
                  mostrar_traza=argumentos.traza, malla=cgra.malla)
    final, _ = cgra.regiones()
    mostrar_resultados(inicial, final, formas)

    contexto = {"programa": str(programa), "formas": formas}
    if argumentos.ciclos:
        mostrar_estadisticas_ciclos(programas, cgra.filas, cgra.columnas, **contexto)
    # El reporte de ciclos se guarda en todas las corridas, en la carpeta del
    # programa y con un solo nombre, que se reescribe en la corrida siguiente.
    ruta_reporte = Path(argumentos.reporte_ciclos) if argumentos.reporte_ciclos else programa / "reporte_ciclos.txt"
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    guardar_reporte_ciclos(programas, cgra.filas, cgra.columnas, ruta_reporte, **contexto)
    print(f"\n  Reporte de ciclos guardado en {ruta_reporte}")
    if argumentos.write_back:
        escribir_memoria(ruta_memoria, final, formas)
        print(f"  Memoria con los resultados guardada en {ruta_memoria}")


def main():
    parser = argparse.ArgumentParser(description="Ejecuta un programa en una CGRA de tamaño NxM.")
    parser.add_argument(
        "programa", help="Carpeta del programa, con pe_instructions/ y memoria.bin (ej. src/matmul)",
    )
    parser.add_argument(
        "--instrucciones", metavar="DIR", help="Usa estos PE*.txt en vez de <programa>/pe_instructions",
    )
    parser.add_argument("--memoria", metavar="RUTA", help="Usa esta memoria en vez de <programa>/memoria.bin")
    parser.add_argument(
        "--cgra", metavar="JSON", default=RUTA_POR_DEFECTO,
        help="Formato de la CGRA: malla, memoria y tipo de cada PE (por defecto src/cgra.json)",
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
    ejecutar_y_mostrar(parser.parse_args())


if __name__ == "__main__":
    main()
