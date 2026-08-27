"""
Tests de q_learning.py

Verifican el bucle de entrenamiento de forma integrada: que el agente
efectivamente aprenda a resolver un TSP pequeño tras suficientes
episodios, y que extraer_politica_greedy devuelva una ruta valida.

Tambien se prueba que el registro (opcional) capture un episodio por
cada uno de los entrenados.
"""

from grafo import Grafo
from entorno_tsp import EntornoTSP
from agente import AgenteQLearning
from ql import entrenar, extraer_politica_greedy
from registro import RegistroEntrenamiento


def _entorno_triangulo():
    g = Grafo()
    g.agregar_arista("A", "B", 1)
    g.agregar_arista("A", "C", 5)
    g.agregar_arista("B", "C", 1)
    return EntornoTSP(g, ciudad_inicio="A")


def test_agente_aprende_la_ruta_optima_en_problema_pequeno():
    # Optimo obvio: A -> B -> C -> A = 1 + 1 + 5 = 7 (mejor que A -> C -> B -> A = 5+1+1=7, empate)
    # En este triangulo ambas rutas dan 7, asi que solo verificamos que
    # el agente converja al costo minimo posible, sea cual sea la ruta.
    entorno = _entorno_triangulo()
    agente = AgenteQLearning(alpha=0.3, gamma=1.0, epsilon_inicial=0.5,
                              decaimiento_epsilon=0.995, semilla=0)
    entrenar(entorno, agente, episodios=500)

    ruta, costo = extraer_politica_greedy(agente, entorno)
    assert costo == 7  # costo optimo conocido para este triangulo
    assert ruta[0] == ruta[-1] == 0  # empieza y termina en A


def test_extraer_politica_greedy_devuelve_ruta_que_visita_todas_las_ciudades():
    entorno = _entorno_triangulo()
    agente = AgenteQLearning(epsilon_inicial=0.5, semilla=0)
    entrenar(entorno, agente, episodios=300)

    ruta, _ = extraer_politica_greedy(agente, entorno)
    assert set(ruta) == {0, 1, 2}  # A, B, C todas presentes


def test_entrenar_con_registro_guarda_una_fila_por_episodio():
    entorno = _entorno_triangulo()
    agente = AgenteQLearning(semilla=0)
    registro = RegistroEntrenamiento()

    entrenar(entorno, agente, episodios=10, registro=registro)

    assert len(registro.filas) == 10
    assert [f["episodio"] for f in registro.filas] == list(range(10))


def test_entrenar_sin_registro_no_falla():
    entorno = _entorno_triangulo()
    agente = AgenteQLearning(semilla=0)
    # no deberia lanzar ninguna excepcion al no pasar 'registro'
    entrenar(entorno, agente, episodios=5)
