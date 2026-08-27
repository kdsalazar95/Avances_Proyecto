"""
Modulo del ENTORNO (interfaz de Reinforcement Learning).

Envuelve el mismo Grafo de ciudades (grafo.py) que usa problema_tsp.py,
pero con una interfaz distinta: la que necesita un agente de RL para
interactuar paso a paso, en vez de la exploracion exhaustiva que usa
Dijkstra.

Interfaz tipo "entorno" (parecida a la de Gymnasium/OpenAI Gym):
    - reset()              -> vuelve al estado inicial, empieza un episodio
    - paso(accion)          -> ejecuta una accion, devuelve
                               (nuevo_estado, recompensa, terminado)
    - acciones_validas(e)   -> que acciones (ciudades) se pueden tomar
                               desde el estado 'e'

El estado sigue siendo (ciudad_actual, mascara_de_visitadas), igual que
en problema_tsp.py -- es la misma idea de estado, solo que aqui el
agente lo descubre interactuando, en vez de que Dijkstra lo explore
todo de una vez.

Recompensa: se usa el negativo de la distancia recorrida en cada paso,
porque un agente de RL maximiza recompensa acumulada, y nosotros
queremos minimizar distancia total.
"""


class EntornoTSP:
    def __init__(self, grafo, ciudad_inicio=None):
        """
        grafo: instancia de Grafo (ver grafo.py), con ciudades y distancias.
        ciudad_inicio: nombre o indice de la ciudad de partida.
        """
        self.grafo = grafo
        self.n = grafo.n
        self.inicio = (
            grafo.indice_de(ciudad_inicio)
            if isinstance(ciudad_inicio, str)
            else (ciudad_inicio if ciudad_inicio is not None else 0)
        )
        self.mascara_completa = (1 << self.n) - 1
        self.estado = None

    def reset(self):
        """Empieza un episodio nuevo: vuelve a la ciudad de inicio."""
        self.estado = (self.inicio, 1 << self.inicio)
        return self.estado

    def acciones_validas(self, estado):
        """
        Ciudades a las que se puede mover el agente desde 'estado'.
        Si ya se visitaron todas, la unica accion valida es volver al inicio
        (para cerrar el ciclo).
        """
        ciudad, mascara = estado
        if mascara == self.mascara_completa:
            return [self.inicio]
        return [c for c in range(self.n) if not (mascara & (1 << c))]

    def paso(self, accion):
        """
        Ejecuta la accion 'accion' (moverse a esa ciudad) desde el estado
        actual del entorno. Devuelve (nuevo_estado, recompensa, terminado).
        """
        ciudad, mascara = self.estado
        distancia = self.grafo.peso(ciudad, accion)
        recompensa = -distancia  # penalizar por distancia recorrida

        if mascara == self.mascara_completa and accion == self.inicio:
            # Se cierra el ciclo: episodio terminado
            nuevo_estado = (accion, mascara)
            terminado = True
        else:
            nueva_mascara = mascara | (1 << accion)
            nuevo_estado = (accion, nueva_mascara)
            terminado = False

        self.estado = nuevo_estado
        return nuevo_estado, recompensa, terminado

    def nombres_de_ruta(self, ruta_indices):
        """Convierte una ruta de indices en nombres legibles, usando el grafo."""
        return [self.grafo.nombre_de(i) for i in ruta_indices]