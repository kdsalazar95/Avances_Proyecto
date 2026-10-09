"""Cargador de programas de la CGRA, el equivalente al cargador del
sistema operativo (execve en Linux).

Recibe una CGRA recién creada (memoria en cero) y un programa compilado
por los skills:

    memoria.bin         imagen de la memoria, generada por el .c
    pe_instructions/    un PE{fila}{columna}.txt por PE

y, antes de que corra el primer ciclo:

1. verifica que la imagen quepa en la memoria de la CGRA,
2. copia la imagen a partir de la dirección ``base`` (el resto de la
   memoria sigue en cero) y registra la tabla de símbolos,
3. verifica que cada PE solo use instrucciones de su tipo (cgra.json),
4. enlaza los LD/ST: cada ``a[3]`` se convierte en la dirección de la
   palabra en la memoria plana (dirección de ``a`` + 3),
5. carga en cada PE su programa.

No se ejecuta ningún algoritmo: solo se copian datos e instrucciones.
"""

from memoria_binaria import leer_imagen
from formato_cgra import validar_tipos
from programas_pe import cargar_programas_pe


def cargar_imagen(cgra, ruta_memoria, base=0):
    """Copia memoria.bin en la memoria de la CGRA a partir de ``base``."""
    palabras, simbolos = leer_imagen(ruta_memoria)
    fin = base + len(palabras)
    if base < 0 or fin > len(cgra.memoria):
        raise ValueError(
            f"{ruta_memoria}: la imagen ocupa {len(palabras)} palabras desde la dirección {base} "
            f"y la memoria de la CGRA tiene {len(cgra.memoria)}"
        )
    cgra.memoria[base:fin] = palabras
    cgra.simbolos = {
        nombre: (base + direccion, filas, columnas)
        for nombre, (direccion, filas, columnas) in simbolos.items()
    }


def enlazar(programas, simbolos):
    """Resuelve el ``banco[índice]`` de cada LD/ST a una dirección absoluta."""
    for pe_id, programa in programas.items():
        for ciclo, instruccion in programa.items():
            if instruccion["op"] not in {"ld", "st"}:
                continue
            banco, indice = instruccion["bank"], instruccion["index"]
            if banco not in simbolos:
                raise ValueError(f"{pe_id}, ciclo {ciclo}: la región '{banco}' no está en la memoria cargada")
            direccion, filas, columnas = simbolos[banco]
            if not 0 <= indice < filas * columnas:
                raise ValueError(
                    f"{pe_id}, ciclo {ciclo}: {banco}[{indice}] se sale de la región ({filas * columnas} palabras)"
                )
            instruccion["addr"] = direccion + indice


def cargar_programa(cgra, ruta_memoria, ruta_instrucciones, base=0):
    """Deja la CGRA lista para ejecutar y devuelve los programas enlazados."""
    cargar_imagen(cgra, ruta_memoria, base)
    programas = cargar_programas_pe(ruta_instrucciones, cgra.filas, cgra.columnas)
    validar_tipos(programas, cgra.formato)
    enlazar(programas, cgra.simbolos)
    for pe_id, programa in programas.items():
        cgra.pe(pe_id).load_instructions([programa[ciclo] for ciclo in sorted(programa)])
    return programas
