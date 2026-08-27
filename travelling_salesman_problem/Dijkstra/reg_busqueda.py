"""
Modulo de REGISTRO (busqueda tipo Dijkstra).

Acumula metricas de cada iteracion de una busqueda de camino minimo
estilo Dijkstra (una expansion de estado a la vez) y las guarda en un
CSV para poder inspeccionar como avanzo la busqueda: cuantos estados
se expandieron, como crecio el costo acumulado, y como crecio/decrecio
la cola de prioridad.

A diferencia de registro.py (usado para RL), aqui NO hay episodios ni
exploracion aleatoria: Dijkstra hace una sola pasada, expandiendo un
estado a la vez, siempre en orden de menor costo acumulado.

No sabe nada de TSP ni de ciudades: el "estado" que registra es el
mismo objeto generico que usa dijkstra.py (una tupla (ciudad, mascara)
en el caso del TSP, pero podria ser cualquier otra cosa en otro problema).
"""

import csv


class RegistroBusqueda:
    def __init__(self):
        self.filas = []

    def registrar(self, iteracion, estado, costo_acumulado, tamano_cola):
        """
        Guarda las metricas de UNA expansion de estado.

        iteracion: numero de estados expandidos hasta el momento (empieza en 0).
        estado: el estado que se acaba de expandir (tal cual lo entrega el problema).
        costo_acumulado: costo total para llegar a ese estado.
        tamano_cola: cuantos elementos quedan en la cola de prioridad tras expandir.
        """
        self.filas.append({
            "iteracion": iteracion,
            "estado": estado,
            "costo_acumulado": costo_acumulado,
            "tamano_cola": tamano_cola,
        })

    def guardar_csv(self, ruta):
        """Escribe todas las filas registradas a un archivo CSV."""
        if not self.filas:
            return
        campos = list(self.filas[0].keys())
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(self.filas)