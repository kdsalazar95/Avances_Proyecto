"""
Modulo del GRAFO.

Estructura de datos pura para un grafo no dirigido con pesos:
    - nodos = ciudades
    - aristas = conexiones directas entre ciudades, con una distancia asociada

Este modulo no sabe nada de TSP, de "estados" ni de Dijkstra. Es solo
la representacion del mapa de ciudades y sus conexiones, reutilizable
para cualquier problema que necesite un grafo de ciudades.
"""


class Grafo:
    def __init__(self):
        self._nodos = []              # lista de nombres de ciudades, ej. ["A", "B", ...]
        self._indice = {}             # nombre -> indice, para busqueda rapida
        self._aristas = {}            # (i, j) -> peso, con i < j (grafo no dirigido)

    def agregar_ciudad(self, nombre):
        """Agrega una ciudad (nodo) al grafo si no existe ya."""
        if nombre not in self._indice:
            self._indice[nombre] = len(self._nodos)
            self._nodos.append(nombre)
        return self._indice[nombre]

    def agregar_arista(self, ciudad_a, ciudad_b, peso):
        """Agrega una conexion (arista) entre dos ciudades con un peso (distancia)."""
        i = self.agregar_ciudad(ciudad_a)
        j = self.agregar_ciudad(ciudad_b)
        clave = (i, j) if i < j else (j, i)
        self._aristas[clave] = peso

    def peso(self, ciudad_a, ciudad_b):
        """Devuelve el peso (distancia) de la arista entre dos ciudades."""
        i = self._indice[ciudad_a] if isinstance(ciudad_a, str) else ciudad_a
        j = self._indice[ciudad_b] if isinstance(ciudad_b, str) else ciudad_b
        clave = (i, j) if i < j else (j, i)
        return self._aristas[clave]

    def vecinos_de(self, ciudad):
        """Devuelve [(ciudad_vecina_idx, peso), ...] para todas las ciudades
        conectadas directamente con 'ciudad' (por indice o nombre)."""
        i = self._indice[ciudad] if isinstance(ciudad, str) else ciudad
        resultado = []
        for (a, b), peso in self._aristas.items():
            if a == i:
                resultado.append((b, peso))
            elif b == i:
                resultado.append((a, peso))
        return resultado

    @property
    def nodos(self):
        """Lista de nombres de ciudades, en orden de indice."""
        return list(self._nodos)

    @property
    def n(self):
        """Cantidad de ciudades (nodos) en el grafo."""
        return len(self._nodos)

    def indice_de(self, nombre):
        return self._indice[nombre]

    def nombre_de(self, indice):
        return self._nodos[indice]