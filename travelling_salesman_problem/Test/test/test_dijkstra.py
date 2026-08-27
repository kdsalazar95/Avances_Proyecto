"""
Tests de dijkstra.py

Primero se prueba el algoritmo de forma aislada, con un problema
generico de juguete (sin ciudades ni TSP), para confirmar que
"dijkstra.py" es realmente independiente del dominio.

Despues se prueba integrado con ProblemaTSP + Grafo, sobre el ejemplo
de 5 ciudades cuyo costo optimo conocido es 79.
"""

from dijkstra import dijkstra, reconstruir_camino
from grafo import Grafo
from problema_tsp import ProblemaTSP
from reg_busqueda import RegistroBusqueda


class ProblemaLinealDeJuguete:
    """
    Problema minimo que no tiene nada que ver con TSP: una linea de
    estados 0 -> 1 -> 2 -> 3, cada paso cuesta 1. Sirve para probar
    dijkstra.py de forma totalmente aislada del dominio del TSP.
    """
    def __init__(self, n_estados=4):
        self.n_estados = n_estados

    def estado_inicial(self):
        return 0

    def vecinos(self, estado):
        if estado + 1 < self.n_estados:
            return [(estado + 1, 1)]
        return []


def test_dijkstra_en_problema_de_juguete_encuentra_costos_correctos():
    problema = ProblemaLinealDeJuguete(n_estados=4)
    costos, padres = dijkstra(problema)
    assert costos == {0: 0, 1: 1, 2: 2, 3: 3}


def test_reconstruir_camino_sigue_los_padres_en_orden():
    problema = ProblemaLinealDeJuguete(n_estados=4)
    costos, padres = dijkstra(problema)
    camino = reconstruir_camino(padres, estado_final=3, estado_inicial=0)
    assert camino == [0, 1, 2, 3]


def test_dijkstra_encuentra_el_optimo_conocido_en_tsp_5_ciudades():
    """
    Mismo ejemplo usado en main.py: la ruta optima conocida tiene
    costo 79 (A-E-D-C-B-A o su reverso).
    """
    grafo = Grafo()
    distancias = {
        ("A", "B"): 29, ("A", "C"): 20, ("A", "D"): 21, ("A", "E"): 16,
        ("B", "C"): 15, ("B", "D"): 29, ("B", "E"): 28,
        ("C", "D"): 15, ("C", "E"): 14,
        ("D", "E"): 4,
    }
    for (a, b), peso in distancias.items():
        grafo.agregar_arista(a, b, peso)

    problema = ProblemaTSP(grafo, ciudad_inicio="A")
    costos, _ = dijkstra(problema)

    mejor_costo = min(
        costo + problema.costo_cierre_ciclo(ciudad)
        for (ciudad, _mascara), costo in costos.items()
        if problema.es_meta((ciudad, _mascara))
    )
    assert mejor_costo == 79


def test_dijkstra_con_registro_guarda_una_fila_por_estado_expandido():
    grafo = Grafo()
    grafo.agregar_arista("A", "B", 1)
    grafo.agregar_arista("A", "C", 2)
    grafo.agregar_arista("B", "C", 3)
    problema = ProblemaTSP(grafo, ciudad_inicio="A")

    registro = RegistroBusqueda()
    dijkstra(problema, registro=registro)

    assert len(registro.filas) > 0
    # las iteraciones deben ser consecutivas empezando en 0
    iteraciones = [fila["iteracion"] for fila in registro.filas]
    assert iteraciones == list(range(len(registro.filas)))
