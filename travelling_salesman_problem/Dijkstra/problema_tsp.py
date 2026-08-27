"""
Modulo del PROBLEMA.

Toma un Grafo de ciudades (nodos + aristas con peso) y lo convierte en
un grafo de ESTADOS para que Dijkstra pueda resolver el TSP.

Por que hace falta esta capa intermedia:
Dijkstra normal, aplicado directamente sobre el grafo de ciudades,
encontraria el camino mas corto entre dos ciudades - pero NO garantiza
pasar por todas las ciudades exactamente una vez. Para lograr eso,
cada "nodo" del grafo que le pasamos a Dijkstra debe ser en realidad
el par (ciudad_actual, conjunto_de_ciudades_ya_visitadas). Este modulo
arma ese grafo de estados usando como base el Grafo de ciudades.

Nota: se asume que el Grafo de ciudades es completo (existe una arista
entre cada par de ciudades), como es habitual en el TSP clasico.
"""


class ProblemaTSP:
    def __init__(self, grafo, ciudad_inicio=None):
        """
        grafo: instancia de Grafo (ver grafo.py), con las ciudades y sus
               distancias ya cargadas.
        ciudad_inicio: nombre o indice de la ciudad de partida (por defecto,
               la primera ciudad agregada al grafo).
        """
        self.grafo = grafo
        self.n = grafo.n
        self.inicio = (
            grafo.indice_de(ciudad_inicio)
            if isinstance(ciudad_inicio, str)
            else (ciudad_inicio if ciudad_inicio is not None else 0)
        )
        self.mascara_completa = (1 << self.n) - 1

    def estado_inicial(self):
        """Estado antes de moverse: en la ciudad de inicio, solo ella visitada."""
        return (self.inicio, 1 << self.inicio)

    def es_meta(self, estado):
        """Un estado es 'meta' si ya se visitaron todas las ciudades."""
        _, mascara = estado
        return mascara == self.mascara_completa

    def vecinos(self, estado):
        """
        Da los estados alcanzables desde 'estado', usando las aristas
        del Grafo de ciudades como costo de cada movimiento.
        Devuelve una lista de tuplas (estado_siguiente, costo_arista).
        """
        ciudad, mascara = estado
        resultado = []
        for siguiente in range(self.n):
            if mascara & (1 << siguiente):
                continue  # ya visitada, no se puede repetir
            nueva_mascara = mascara | (1 << siguiente)
            costo = self.grafo.peso(ciudad, siguiente)  # arista del grafo original
            resultado.append(((siguiente, nueva_mascara), costo))
        return resultado

    def costo_cierre_ciclo(self, ciudad):
        """Costo de volver de 'ciudad' al punto de partida, usando la arista del grafo."""
        return self.grafo.peso(ciudad, self.inicio)

    def nombres_de_ruta(self, ruta_indices):
        """Convierte una ruta de indices en nombres legibles, usando el grafo."""
        return [self.grafo.nombre_de(i) for i in ruta_indices]