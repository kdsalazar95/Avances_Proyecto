"""Cargador de programas de la CGRA, el equivalente al cargador del
sistema operativo (execve en Linux).

Recibe una CGRA recién creada (memoria en cero) y un programa compilado
por los skills:

    memoria.bin         imagen de la memoria, generada por operaciones.c
    pe_instructions/    un PE{fila}{columna}.txt por PE

y, antes de que corra el primer ciclo:

1. verifica que la imagen quepa en la memoria de la CGRA,
2. copia la imagen a partir de la dirección ``base`` (el resto de la
   memoria sigue en cero) y registra la tabla de símbolos,
3. lee los programas de los PEs y enlaza sus LD/ST: cada ``a[3]`` se
   convierte en la dirección de la palabra en la memoria plana
   (dirección de ``a`` + 3).

No se ejecuta ningún algoritmo: solo se copian datos y se preparan
instrucciones.
"""

from memoria import ImagenMemoria
from programa import Programa
from formato_cgra import validar_tipos


class Cargador:
    """Deja una CGRA lista para ejecutar un programa compilado."""

    def __init__(self, cgra):
        self.cgra = cgra

    def cargar_imagen(self, ruta_memoria, base=0):
        """Copia memoria.bin en la memoria de la CGRA a partir de ``base``."""
        try:
            self.cgra.memoria.cargar(ImagenMemoria.leer(ruta_memoria), base)
        except ValueError as error:
            raise ValueError(f"{ruta_memoria}: {error}") from None

    def cargar(self, ruta_memoria, ruta_instrucciones, base=0):
        """Carga la imagen y devuelve el ``Programa`` ya enlazado."""
        self.cargar_imagen(ruta_memoria, base)
        programa = Programa.leer(ruta_instrucciones, self.cgra.malla)
        validar_tipos(programa, self.cgra.formato)
        programa.enlazar(self.cgra.memoria.simbolos, self.cgra.formato.ancho_vector)
        return programa
