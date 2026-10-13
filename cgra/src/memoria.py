"""La memoria de la CGRA: el archivo memoria.bin y la memoria de datos.

``memoria.bin`` no lo arma Python: lo genera ``operaciones/operaciones.c``
al ejecutarse (``inicializar_<operacion>()``, con ``guardar_memoria()``
de ``operaciones/memoria_cgra.h``), igual que la memoria de un
procesador. Formato (little-endian), el mismo que describe
``memoria_cgra.h``:

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
"""

import struct
from pathlib import Path

FIRMA = b"CGRAMEM\0"
CABECERA = struct.Struct("<8sII")
SIMBOLO = struct.Struct("<16sIII")


def _extraer_region(palabras, simbolo):
    """Copia los valores de una región según su símbolo."""
    direccion, filas, columnas = simbolo
    return palabras[direccion:direccion + filas * columnas]


def _extraer_regiones(palabras, simbolos):
    """Obtiene los valores y las formas a partir de palabras y símbolos."""
    regiones = {
        nombre: _extraer_region(palabras, simbolo)
        for nombre, simbolo in simbolos.items()
    }
    formas = {
        nombre: (filas, columnas)
        for nombre, (_, filas, columnas) in simbolos.items()
    }
    return regiones, formas


class ImagenMemoria:
    """El contenido de un memoria.bin: palabras y tabla de símbolos.

    Es lo que el cargador copia a la memoria de la CGRA, como el sistema
    operativo copia las secciones de un ejecutable.
    """

    def __init__(self, palabras, simbolos):
        """``simbolos``: ``{nombre: (direccion, filas, columnas)}``."""
        self.palabras = palabras
        self.simbolos = simbolos

    @classmethod
    def leer(cls, ruta):
        """Lee y valida un memoria.bin."""
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

        valores = list(struct.unpack_from(f"<{palabras}f", datos, inicio_datos))
        simbolos = {}
        for r in range(cantidad):
            nombre, direccion, filas, columnas = SIMBOLO.unpack_from(datos, CABECERA.size + r * SIMBOLO.size)
            nombre = nombre.split(b"\0", 1)[0].decode("ascii")
            if direccion + filas * columnas > palabras:
                raise ValueError(f"{ruta}: la región '{nombre}' se sale de la memoria")
            simbolos[nombre] = (direccion, filas, columnas)
        return cls(valores, simbolos)

    @classmethod
    def desde_regiones(cls, regiones, formas):
        """Arma una imagen con las regiones una tras otra.

        ``regiones``: ``{nombre: [valores]}``; ``formas``: ``{nombre: (filas, columnas)}``.
        """
        palabras, simbolos = [], {}
        for nombre, valores in regiones.items():
            filas, columnas = formas[nombre]
            if filas * columnas != len(valores):
                raise ValueError(f"La región '{nombre}' tiene {len(valores)} valores y su forma es {filas}x{columnas}")
            simbolos[nombre] = (len(palabras), filas, columnas)
            palabras.extend(valores)
        return cls(palabras, simbolos)

    def guardar(self, ruta):
        """Escribe la imagen en el formato de memoria.bin."""
        contenido = CABECERA.pack(FIRMA, len(self.simbolos), len(self.palabras))
        for nombre, (direccion, filas, columnas) in self.simbolos.items():
            contenido += SIMBOLO.pack(nombre.encode("ascii"), direccion, filas, columnas)
        contenido += struct.pack(f"<{len(self.palabras)}f", *self.palabras)
        Path(ruta).write_bytes(contenido)
        return Path(ruta)

    def regiones(self):
        """``(regiones, formas)``: los valores y la forma de cada región."""
        return _extraer_regiones(self.palabras, self.simbolos)


class Memoria:
    """La memoria de datos de la CGRA: una lista plana de palabras.

    Al encender la CGRA está toda en cero y no tiene regiones. El
    cargador copia en ella una ``ImagenMemoria`` y registra sus símbolos.
    """

    def __init__(self, palabras):
        self.palabras = [0] * palabras
        # nombre -> (dirección, filas, columnas) de lo que está cargado.
        self.simbolos = {}

    def __len__(self):
        return len(self.palabras)

    def cargar(self, imagen, base=0):
        """Copia ``imagen`` a partir de la dirección ``base``; el resto queda igual."""
        fin = base + len(imagen.palabras)
        if base < 0 or fin > len(self.palabras):
            raise ValueError(
                f"la imagen ocupa {len(imagen.palabras)} palabras desde la dirección {base} "
                f"y la memoria de la CGRA tiene {len(self.palabras)}"
            )
        self.palabras[base:fin] = imagen.palabras
        self.simbolos = {
            nombre: (base + direccion, filas, columnas)
            for nombre, (direccion, filas, columnas) in imagen.simbolos.items()
        }

    def region(self, nombre):
        """Valores actuales de una región cargada."""
        return _extraer_region(self.palabras, self.simbolos[nombre])

    def regiones(self):
        """``(regiones, formas)`` de todo lo cargado, en el orden de la imagen."""
        return _extraer_regiones(self.palabras, self.simbolos)
