"""Presentación en la terminal: dar formato a números e imprimir títulos,
cajas de texto y vectores. No tiene ninguna lógica propia de la CGRA."""


class Consola:
    """Funciones de impresión que comparten el simulador y sus reportes."""

    @staticmethod
    def numero(valor):
        """Da formato compacto a enteros y flotantes (``2.0`` -> ``2``)."""
        return f"{round(valor, 6):g}" if isinstance(valor, (int, float)) else str(valor)

    @staticmethod
    def lista(valores):
        """``[1, 2, 3]`` con cada valor en formato compacto."""
        return "[" + ", ".join(Consola.numero(valor) for valor in valores) + "]"

    @staticmethod
    def titulo(texto):
        """Imprime un título dentro de una caja."""
        ancho = len(texto) + 2
        print()
        print("┌" + "─" * ancho + "┐")
        print(f"│ {texto} │")
        print("└" + "─" * ancho + "┘")

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
        textos = [Consola.numero(valor) for valor in valores]
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
