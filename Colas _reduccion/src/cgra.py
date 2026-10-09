"""El hardware de la CGRA: la malla de PEs y su memoria de datos.

Se construye a partir de su formato (formato_cgra.py / cgra.json): tamaño
de la malla, tamaño de la memoria y tipo de cada PE. Al crearse, la CGRA
está como recién encendida: la memoria está toda en cero, los PEs tienen
sus registros en cero y no tienen ningún programa. No sabe nada de
reducción, matmul ni convolución; para correr algo, el cargador (cargador.py) copia en ella la
imagen de memoria y los programas de los PEs, como hace el sistema
operativo con un ejecutable antes de arrancarlo.
"""

from pe_malla import crear_malla, conectar_malla
from programas_pe import descomponer_pe_id, generar_pe_ids


class CGRA:
    """Malla filas x columnas de PEs con una memoria de datos compartida."""

    def __init__(self, formato):
        """Crea la CGRA "en blanco" según su ``FormatoCGRA``."""
        self.formato = formato
        self.filas = formato.filas
        self.columnas = formato.columnas
        self.malla = crear_malla(self.filas, self.columnas)
        conectar_malla(self.malla)
        for pe_id in self.pe_ids():
            pe = self.pe(pe_id)
            pe.tipo = formato.nombre_tipo(pe_id)
            pe.instrucciones_permitidas = formato.instrucciones(pe_id)
        self.memoria = [0] * formato.palabras_memoria
        # Tabla de símbolos de lo que está cargado:
        # nombre -> (dirección, filas, columnas). Vacía hasta que el
        # cargador copia una imagen.
        self.simbolos = {}

    def pe(self, pe_id):
        """Devuelve el PE con ese ID (por ejemplo ``PE10``)."""
        fila, columna = descomponer_pe_id(pe_id)
        return self.malla[fila][columna]

    def pe_ids(self):
        return generar_pe_ids(self.filas, self.columnas)

    def region(self, nombre):
        """Valores actuales de una región cargada, leídos de la memoria plana."""
        direccion, filas, columnas = self.simbolos[nombre]
        return self.memoria[direccion:direccion + filas * columnas]

    def regiones(self):
        """``(memoria, formas)`` por banco, en el orden de la tabla de símbolos;
        es lo que usan mostrar_resultados() y escribir_memoria()."""
        memoria = {nombre: self.region(nombre) for nombre in self.simbolos}
        formas = {nombre: (filas, columnas) for nombre, (_, filas, columnas) in self.simbolos.items()}
        return memoria, formas
