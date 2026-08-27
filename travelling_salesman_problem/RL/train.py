"""
TRAIN — script de entrenamiento (version Reinforcement Learning).

Carga los hiperparametros desde config_agente.json, arma el mismo Grafo
de ciudades que usa main.py, crea el agente y el entorno, entrena, y
guarda los resultados en disco:
    - historial_entrenamiento.csv  (metricas episodio a episodio)

Para ver el progreso del entrenamiento graficado, correr despues:
    python3 plot.py
(lee el CSV generado aqui; no vuelve a entrenar nada).
"""

import json
import os
import sys

# grafo.py vive en la carpeta hermana 'comun/', compartida con Dijkstra/.
# Hay que agregarla al path antes de poder importarla.
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_AQUI, "..", "comun"))

from grafo import Grafo
from entorno_tsp import EntornoTSP
from agente import AgenteQLearning
from ql import entrenar, extraer_politica_greedy
from registro import RegistroEntrenamiento


if __name__ == "__main__":
    # 1. Cargar hiperparametros desde el JSON (no hay nada "hardcodeado" aqui)
    #    Ruta construida a partir de la ubicacion del propio script, para que
    #    funcione sin importar desde que carpeta se ejecute train.py.
    with open(os.path.join(_AQUI, "config_agente.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Construir el grafo de ciudades: nodos + aristas con peso (distancia)
    #    (identico al de main.py, para poder comparar resultados)
    grafo = Grafo()

    distancias = {
        ("A", "B"): 29, ("A", "C"): 20, ("A", "D"): 21, ("A", "E"): 16,
        ("B", "C"): 15, ("B", "D"): 29, ("B", "E"): 28,
        ("C", "D"): 15, ("C", "E"): 14,
        ("D", "E"): 4,
    }
    for (ciudad_a, ciudad_b), peso in distancias.items():
        grafo.agregar_arista(ciudad_a, ciudad_b, peso)

    # 3. Envolver el grafo en el entorno de RL
    entorno = EntornoTSP(grafo, ciudad_inicio="A")

    # 4. Crear el agente a partir de los hiperparametros del JSON
    config_agente = dict(config["agente"])
    config_agente["semilla"] = config["entrenamiento"].get("semilla")
    agente = AgenteQLearning.desde_config(config_agente)

    # 5. Entrenar: hacer que el agente juegue episodios contra el entorno,
    #    registrando las metricas de cada episodio
    episodios = config["entrenamiento"]["episodios"]
    registro = RegistroEntrenamiento()
    entrenar(entorno, agente, episodios, registro=registro)

    # 6. Guardar el registro completo en un CSV (misma logica: ruta absoluta
    #    relativa al script, no al directorio desde donde se ejecuta)
    registro.guardar_csv(os.path.join(_AQUI, "historial_entrenamiento.csv"))

    # 7. Extraer la mejor ruta segun lo aprendido (sin exploracion)
    ruta_indices, costo = extraer_politica_greedy(agente, entorno)

    print("Ruta aprendida:", " -> ".join(entorno.nombres_de_ruta(ruta_indices)))
    print("Costo total:", costo)
    print()
    print("Progreso del entrenamiento (cada 500 episodios):")
    print(f"{'episodio':>10} {'costo_total':>12} {'pasos':>7} {'epsilon':>9}")
    for fila in registro.resumen(cada_n=500):
        print(f"{fila['episodio']:>10} {fila['costo_total']:>12.1f} "
              f"{fila['pasos']:>7} {fila['epsilon']:>9.4f}")