"""
Modulo de EJECUCION.

Arma el Grafo de ciudades, lo envuelve en un ProblemaTSP (que construye
el grafo de estados), corre el algoritmo de Dijkstra generico sobre ese
grafo de estados, cierra el ciclo (regreso al origen), muestra la ruta
optima encontrada y guarda un registro de la busqueda en CSV.

A diferencia de train.py (RL), aqui NO hay entrenamiento: Dijkstra hace
una sola pasada exacta sobre el grafo de estados. El registro sirve para
ver cuantos estados exploro y como crecio el costo acumulado durante
esa unica pasada, no para comparar entre "episodios".
"""

import os
import sys
import time

# grafo.py vive en la carpeta hermana 'comun/', compartida con RL/.
# Hay que agregarla al path antes de poder importarla.
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_AQUI, "..", "comun"))

from grafo import Grafo
from problema_tsp import ProblemaTSP
from dijkstra import dijkstra, reconstruir_camino
from reg_busqueda import RegistroBusqueda


def resolver_tsp(problema, registro=None):
    """
    Corre Dijkstra sobre el grafo de estados del problema y arma
    la ruta final, incluyendo el regreso a la ciudad de inicio.
    """
    costos, padres = dijkstra(problema, registro=registro)
    inicio = problema.estado_inicial()

    # Buscar, entre todos los estados "meta" (todas las ciudades visitadas),
    # cual da el menor costo total al cerrar el ciclo de regreso al origen.
    mejor_costo = float("inf")
    mejor_estado_final = None

    for estado, costo in costos.items():
        if problema.es_meta(estado):
            ciudad_actual, _ = estado
            costo_total = costo + problema.costo_cierre_ciclo(ciudad_actual)
            if costo_total < mejor_costo:
                mejor_costo = costo_total
                mejor_estado_final = estado

    # Reconstruir la secuencia de estados y extraer solo las ciudades
    camino_estados = reconstruir_camino(padres, mejor_estado_final, inicio)
    ruta_indices = [ciudad for ciudad, _mascara in camino_estados]
    ruta_indices.append(problema.inicio)  # cerrar el ciclo visualmente

    return mejor_costo, ruta_indices


if __name__ == "__main__":
    # 1. Construir el grafo de ciudades: nodos + aristas con peso (distancia)
    grafo = Grafo()

    distancias = {
        ("A", "B"): 29, ("A", "C"): 20, ("A", "D"): 21, ("A", "E"): 16,
        ("B", "C"): 15, ("B", "D"): 29, ("B", "E"): 28,
        ("C", "D"): 15, ("C", "E"): 14,
        ("D", "E"): 4,
    }
    for (ciudad_a, ciudad_b), peso in distancias.items():
        grafo.agregar_arista(ciudad_a, ciudad_b, peso)

    # 2. Envolver el grafo en el problema TSP
    problema = ProblemaTSP(grafo, ciudad_inicio="A")

    # 3. Resolver con Dijkstra sobre el grafo de estados, registrando cada
    #    estado que se expande durante la busqueda
    registro = RegistroBusqueda()
    inicio_reloj = time.perf_counter()
    costo, ruta = resolver_tsp(problema, registro=registro)
    tiempo_total = time.perf_counter() - inicio_reloj

    # 4. Guardar el registro de la busqueda en CSV (ruta absoluta relativa
    #    al script, no al directorio desde donde se ejecuta)
    registro.guardar_csv(os.path.join(_AQUI, "historial_busqueda_dijkstra.csv"))

    print("Ruta optima:", " -> ".join(problema.nombres_de_ruta(ruta)))
    print("Costo total:", costo)
    print("Estados expandidos:", len(registro.filas))
    print(f"Tiempo de ejecucion: {tiempo_total * 1000:.2f} ms")