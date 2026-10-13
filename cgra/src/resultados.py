"""Presentación del resultado de una corrida de la CGRA.

La CGRA no sabe qué operación ejecutó: igual que un procesador, solo
ejecuta los programas de sus PEs sobre la memoria. Por eso el resultado
se muestra a partir de la memoria misma: las regiones que el programa
modificó, cada una con la forma que declara memoria.bin (un escalar, un
vector o una matriz).
"""

from consola import Consola
from formato_numero import formatear_numero


class Resultados:
    """Compara la memoria antes y después de ejecutar."""

    def __init__(self, inicial, final, formas):
        """``inicial``/``final``: ``{región: [valores]}``; ``formas``: ``{región: (filas, columnas)}``."""
        self.inicial = inicial
        self.final = final
        self.formas = formas

    def modificadas(self):
        """Nombres de las regiones cuyo contenido cambió."""
        return [nombre for nombre, valores in self.final.items() if valores != self.inicial.get(nombre)]

    def mostrar(self):
        """Muestra cada región modificada según su forma."""
        modificadas = self.modificadas()
        if not modificadas:
            Consola.titulo("RESULTADO")
            Consola.caja(["Ejecución terminada: los PEs no modificaron la memoria."])
            return
        for nombre in modificadas:
            self._mostrar_region(nombre, self.final[nombre], *self.formas[nombre])

    @staticmethod
    def _mostrar_region(nombre, valores, filas, columnas):
        if filas * columnas == 1:
            Consola.titulo(f"RESULTADO: {nombre}")
            Consola.caja([f"{nombre} = {formatear_numero(valores[0])}"])
        elif filas == 1 or columnas == 1:
            Consola.vector(nombre, valores)
        else:
            Consola.matriz(f"RESULTADO: {nombre} ({filas}x{columnas})", valores, filas, columnas)
