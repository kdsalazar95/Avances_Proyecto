"""Lee y escribe la memoria binaria de la CGRA (memoria.bin).

El archivo no lo arma Python: lo genera el propio programa C al
ejecutarse (``inicializar_memoria()`` de cada .c, con ``guardar_memoria()``
de ``memoria_cgra.h``), igual que la memoria de un procesador. Aquí solo
se carga para que los PEs la usen y, con ``--write-back``, se vuelve a
guardar con los resultados.

Formato (little-endian), el mismo que describe ``memoria_cgra.h``:

    char     firma[8]        "CGRAMEM\\0"
    uint32   cantidad        número de regiones
    uint32   palabras        tamaño total de la memoria, en floats
    cantidad x {             tabla de símbolos
      char   nombre[16]      banco que usan los LD/ST de los PE*.txt
      uint32 direccion       primera palabra de la región
      uint32 filas
      uint32 columnas
    }
    float32  datos[palabras] memoria plana; las matrices, fila por fila

Cada región se entrega como un banco ``{nombre: [valores]}``, que es lo
que indexan ``LD``/``ST`` (``a[3]``). La forma (filas x columnas) de cada
región sirve para mostrar el resultado sin saber qué operación se corrió.
"""

import struct
from pathlib import Path

FIRMA = b"CGRAMEM\0"
CABECERA = struct.Struct("<8sII")
SIMBOLO = struct.Struct("<16sIII")


def leer_imagen(ruta):
    """Lee memoria.bin tal como está en el archivo: devuelve ``(palabras,
    simbolos)``, con la memoria plana y, por cada región, su
    ``(direccion, filas, columnas)`` dentro de esa memoria.

    Es la imagen que el cargador copia a la memoria de la CGRA, como el
    sistema operativo copia las secciones de un ejecutable."""
    datos = Path(ruta).read_bytes()
    if len(datos) < CABECERA.size:
        raise ValueError(f"{ruta}: archivo demasiado corto para ser una memoria de la CGRA")
    firma, cantidad, palabras = CABECERA.unpack_from(datos)
    if firma != FIRMA:
        raise ValueError(f"{ruta}: no es una memoria de la CGRA (firma {firma!r})")
    inicio_datos = CABECERA.size + cantidad * SIMBOLO.size
    esperado = inicio_datos + 4 * palabras
    if len(datos) != esperado:
        raise ValueError(f"{ruta}: se esperaban {esperado} bytes y tiene {len(datos)}")

    valores = [numero(valor) for valor in struct.unpack_from(f"<{palabras}f", datos, inicio_datos)]
    simbolos = {}
    for r in range(cantidad):
        nombre, direccion, filas, columnas = SIMBOLO.unpack_from(datos, CABECERA.size + r * SIMBOLO.size)
        nombre = nombre.split(b"\0", 1)[0].decode("ascii")
        if direccion + filas * columnas > palabras:
            raise ValueError(f"{ruta}: la región '{nombre}' se sale de la memoria")
        simbolos[nombre] = (direccion, filas, columnas)
    return valores, simbolos


def cargar_memoria(ruta):
    """Devuelve ``(memoria, formas)``: los bancos con sus valores y la
    forma ``(filas, columnas)`` de cada uno, en el orden del archivo."""
    valores, simbolos = leer_imagen(ruta)
    memoria, formas = {}, {}
    for nombre, (direccion, filas, columnas) in simbolos.items():
        memoria[nombre] = valores[direccion:direccion + filas * columnas]
        formas[nombre] = (filas, columnas)
    return memoria, formas


def numero(valor):
    """Deja como int los floats enteros, como hace ``MOV`` con sus constantes."""
    return int(valor) if valor.is_integer() else valor


def escribir_memoria(ruta, memoria, formas):
    """Guarda ``memoria`` en el mismo formato, con las regiones una tras otra."""
    simbolos, datos = [], []
    for nombre, valores in memoria.items():
        filas, columnas = formas[nombre]
        if filas * columnas != len(valores):
            raise ValueError(f"La región '{nombre}' tiene {len(valores)} valores y su forma es {filas}x{columnas}")
        simbolos.append(SIMBOLO.pack(nombre.encode("ascii"), len(datos), filas, columnas))
        datos.extend(valores)
    contenido = CABECERA.pack(FIRMA, len(simbolos), len(datos)) + b"".join(simbolos)
    contenido += struct.pack(f"<{len(datos)}f", *datos)
    Path(ruta).write_bytes(contenido)
    return Path(ruta)
