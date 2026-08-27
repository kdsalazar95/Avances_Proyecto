"""
Modulo de REGISTRO.

Acumula metricas del entrenamiento, episodio a episodio, y las guarda
en un archivo CSV para poder inspeccionarlas o graficarlas despues
(ej. abrir el CSV en Excel, o cargarlo con pandas/matplotlib).

No sabe nada de TSP ni de Q-learning: solo recibe numeros y los guarda.
Se le podria usar para registrar el entrenamiento de cualquier otro
agente, no solo el de este proyecto.
"""

import csv


class RegistroEntrenamiento:
    def __init__(self):
        self.filas = []

    def registrar(self, episodio, recompensa_total, costo_total, pasos, epsilon):
        """
        Guarda las metricas de UN episodio de entrenamiento.

        episodio: numero de episodio (empezando en 0 o 1).
        recompensa_total: suma de recompensas obtenidas en el episodio
                           (sera negativa, ya que la recompensa es -distancia).
        costo_total: distancia total recorrida en el episodio (positivo).
        pasos: cantidad de movimientos que hizo el agente en el episodio.
        epsilon: valor de epsilon (exploracion) usado durante ese episodio.
        """
        self.filas.append({
            "episodio": episodio,
            "recompensa_total": recompensa_total,
            "costo_total": costo_total,
            "pasos": pasos,
            "epsilon": epsilon,
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

    def resumen(self, cada_n=500):
        """
        Devuelve una lista de filas espaciadas cada 'cada_n' episodios,
        util para imprimir un resumen corto del progreso en consola
        sin mostrar los miles de episodios uno por uno.
        """
        return [fila for fila in self.filas if fila["episodio"] % cada_n == 0]