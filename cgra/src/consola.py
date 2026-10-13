"""Presentación en la terminal: imprimir títulos, cajas y datos."""

from formato_numero import formatear_numero


class Consola:
    """Funciones de impresión que comparten el simulador y sus reportes."""

    @staticmethod
    def lista(valores):
        """Convierte una secuencia en una lista de texto."""
        return "[" + ", ".join(formatear_numero(valor) for valor in valores) + "]"

    @staticmethod
    def titulo(texto):
        """Imprime un título dentro de una caja."""
        print()
        Consola.caja([texto])

    @staticmethod
    def caja(lineas):
        """Imprime varias líneas dentro de una caja de texto."""
        ancho = max(len(linea) for linea in lineas) + 2
        print("┌" + "─" * ancho + "┐")
        for linea in lineas:
            print(f"│ {linea.ljust(ancho - 1)}│")
        print("└" + "─" * ancho + "┘")

    @staticmethod
    def vector(nombre, valores, por_fila=10):
        """Imprime un vector en filas cortas, con el índice inicial de cada fila.

        Los valores van alineados a la derecha, en columnas parejas, para que
        un vector largo (por ejemplo los 100 elementos de una reducción) se
        pueda leer y ubicar por posición.
        """
        Consola.titulo(f"{nombre}: {len(valores)} valores")
        textos = [formatear_numero(valor) for valor in valores]
        ancho_valor = max([len(texto) for texto in textos] + [3])
        ancho_indice = len(str(max(len(valores) - 1, 0)))
        for inicio in range(0, len(textos), por_fila):
            fila = textos[inicio:inicio + por_fila]
            print(f"  [{inicio:>{ancho_indice}}] " + " ".join(texto.rjust(ancho_valor) for texto in fila))

    @staticmethod
    def matriz(titulo, valores, filas, columnas):
        """Imprime una lista aplanada (fila por fila) como una matriz."""
        Consola.titulo(titulo)
        for f in range(filas):
            print("  " + Consola.lista(valores[f * columnas:(f + 1) * columnas]))
