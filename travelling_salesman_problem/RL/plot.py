"""
PLOT — script de visualizacion.

Lee el archivo historial_entrenamiento.csv (generado por train.py al
entrenar) y produce graficas del progreso del entrenamiento:
    - costo total por episodio (para ver si el agente va mejorando)
    - epsilon a lo largo del entrenamiento (para ver como decae la exploracion)

No entrena nada ni sabe de TSP: solo lee un CSV con columnas
episodio, recompensa_total, costo_total, pasos, epsilon y las grafica.
Se puede correr las veces que quieras sobre el mismo CSV, sin volver
a ejecutar train.py.
"""

import csv
import os
import matplotlib.pyplot as plt


def leer_historial(ruta_csv):
    """Lee el CSV y devuelve listas separadas por columna."""
    episodios, costos, epsilons = [], [], []
    with open(ruta_csv, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            episodios.append(int(fila["episodio"]))
            costos.append(float(fila["costo_total"]))
            epsilons.append(float(fila["epsilon"]))
    return episodios, costos, epsilons


def promedio_movil(valores, ventana=50):
    """
    Suaviza una serie de valores con un promedio movil, para que la
    tendencia general se vea mas clara entre el ruido episodio a episodio.
    """
    suavizado = []
    for i in range(len(valores)):
        inicio = max(0, i - ventana + 1)
        trozo = valores[inicio:i + 1]
        suavizado.append(sum(trozo) / len(trozo))
    return suavizado


def graficar(ruta_csv, ruta_salida="progreso_entrenamiento.png"):
    episodios, costos, epsilons = leer_historial(ruta_csv)
    costos_suavizados = promedio_movil(costos, ventana=50)

    fig, (ax_costo, ax_epsilon) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Grafica de costo por episodio
    ax_costo.plot(episodios, costos, color="#a8d5c8", linewidth=0.8, label="Costo por episodio")
    ax_costo.plot(episodios, costos_suavizados, color="#0F6E56", linewidth=2,
                  label="Promedio movil (50 episodios)")
    ax_costo.set_ylabel("Costo total (distancia)")
    ax_costo.set_title("Progreso del entrenamiento (Q-learning)")
    ax_costo.legend()
    ax_costo.grid(alpha=0.3)

    # Grafica de epsilon (exploracion) a lo largo del entrenamiento
    ax_epsilon.plot(episodios, epsilons, color="#c0392b", linewidth=1.5)
    ax_epsilon.set_xlabel("Episodio")
    ax_epsilon.set_ylabel("Epsilon (exploracion)")
    ax_epsilon.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    print(f"Grafica guardada en: {ruta_salida}")


if __name__ == "__main__":
    # Rutas absolutas relativas a este script, para que funcione sin
    # importar desde que carpeta se ejecute plot.py.
    aqui = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(aqui, "historial_entrenamiento.csv")
    ruta_salida = os.path.join(aqui, "progreso_entrenamiento.png")

    graficar(ruta_csv, ruta_salida)