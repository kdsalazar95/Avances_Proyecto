"""El hardware de la CGRA: la malla de PEs y su memoria de datos.

Se construye a partir de su formato (formato_cgra.py / cgra.json). Al
crearse, la CGRA está como recién encendida: la memoria está toda en
cero y los PEs tienen sus registros en cero. No sabe nada de reducción,
matmul ni convolución: el cargador (cargador.py) copia en ella la imagen
de memoria y prepara el programa, y ``ejecutar()`` corre sus ciclos,
como un procesador con un ejecutable.
"""

from malla import Malla
from memoria import Memoria


class CGRA:
    """Malla filas x columnas de PEs con una memoria de datos compartida."""

    def __init__(self, formato):
        """Crea la CGRA "en blanco" según su ``FormatoCGRA``."""
        self.formato = formato
        self.malla = Malla(formato.filas, formato.columnas)
        for pe_id in self.malla.ids():
            pe = self.malla.pe(pe_id)
            pe.tipo = formato.nombre_tipo(pe_id)
            pe.instrucciones_permitidas = formato.instrucciones(pe_id)
        self.memoria = Memoria(formato.palabras_memoria)

    @property
    def filas(self):
        return self.malla.filas

    @property
    def columnas(self):
        return self.malla.columnas

    def ejecutar(self, programa, traza=None):
        """Ejecuta el programa: en cada ciclo, cada PE ejecuta su instrucción.

        Los SEND de un ciclo se ejecutan antes que el resto, para que el
        RECV del vecino encuentre el dato en la cola en ese mismo ciclo.
        Si se da una ``Traza``, se anota en ella cada instrucción ejecutada.
        """
        for ciclo in programa.ciclos():
            del_ciclo = programa.del_ciclo(ciclo)
            envios = [pe_id for pe_id, instruccion in del_ciclo.items() if instruccion["op"] == "send"]
            resto = [pe_id for pe_id in del_ciclo if pe_id not in envios]
            for pe_id in envios + resto:
                pe = self.malla.pe(pe_id)
                pe.ejecutar(del_ciclo[pe_id], self.memoria.palabras)
                if traza is not None:
                    traza.registrar(ciclo, pe, del_ciclo[pe_id], self.memoria.palabras)
        if traza is not None:
            # La traza queda en el orden de la malla dentro de cada ciclo.
            orden = {pe_id: i for i, pe_id in enumerate(self.malla.ids())}
            traza.entradas.sort(key=lambda e: (e["ciclo"], orden[e["pe"]]))
