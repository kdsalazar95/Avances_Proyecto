"""
Tests de entorno_tsp.py

Verifican la interfaz de entorno tipo RL: reset, acciones_validas y
paso, incluyendo el caso especial de cerrar el ciclo (volver al inicio
tras visitar todas las ciudades).
"""

from grafo import Grafo
from entorno_tsp import EntornoTSP


def _grafo_triangulo():
    g = Grafo()
    g.agregar_arista("A", "B", 1)
    g.agregar_arista("A", "C", 2)
    g.agregar_arista("B", "C", 3)
    return g


def test_reset_vuelve_al_estado_inicial():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    estado = entorno.reset()
    assert estado == (0, 0b001)


def test_acciones_validas_excluye_visitadas():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    entorno.reset()
    acciones = entorno.acciones_validas((0, 0b001))
    assert set(acciones) == {1, 2}  # B y C


def test_acciones_validas_solo_permite_volver_al_inicio_si_ya_se_visito_todo():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    acciones = entorno.acciones_validas((1, 0b111))  # todas visitadas
    assert acciones == [0]  # solo volver a A


def test_paso_devuelve_recompensa_negativa_igual_a_la_distancia():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    entorno.reset()
    nuevo_estado, recompensa, terminado = entorno.paso(1)  # ir a B
    assert recompensa == -1  # peso A-B
    assert terminado is False
    assert nuevo_estado == (1, 0b011)


def test_paso_termina_el_episodio_al_cerrar_el_ciclo():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    entorno.reset()
    entorno.paso(1)  # A -> B
    _, _, terminado_antes = entorno.paso(2)  # B -> C (aun no volvio a A)
    assert terminado_antes is False

    _, _, terminado_final = entorno.paso(0)  # C -> A, cierra el ciclo
    assert terminado_final is True


def test_nombres_de_ruta_traduce_indices_a_nombres():
    entorno = EntornoTSP(_grafo_triangulo(), ciudad_inicio="A")
    assert entorno.nombres_de_ruta([0, 1, 2]) == ["A", "B", "C"]
