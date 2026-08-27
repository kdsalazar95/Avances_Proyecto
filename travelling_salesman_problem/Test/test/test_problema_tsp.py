"""
Tests de problema_tsp.py

Verifican que ProblemaTSP construya correctamente el grafo de estados
(ciudad, mascara_visitadas) a partir de un Grafo, incluyendo el estado
inicial, la deteccion de meta, y los vecinos validos.
"""

from grafo import Grafo
from problema_tsp import ProblemaTSP


def _grafo_triangulo():
    """Grafo simple de 3 ciudades, todas conectadas entre si."""
    g = Grafo()
    g.agregar_arista("A", "B", 1)
    g.agregar_arista("A", "C", 2)
    g.agregar_arista("B", "C", 3)
    return g


def test_estado_inicial_solo_marca_la_ciudad_de_inicio():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    ciudad, mascara = problema.estado_inicial()
    assert ciudad == 0  # indice de "A"
    assert mascara == 0b001  # solo A visitada


def test_es_meta_cuando_todas_las_ciudades_estan_visitadas():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    assert problema.es_meta((1, 0b111)) is True
    assert problema.es_meta((1, 0b011)) is False


def test_vecinos_excluye_ciudades_ya_visitadas():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    estado = (0, 0b001)  # en A, solo A visitada
    vecinos = problema.vecinos(estado)
    ciudades_alcanzables = {v[0][0] for v in vecinos}
    assert ciudades_alcanzables == {1, 2}  # B y C, no A


def test_vecinos_usa_el_peso_correcto_del_grafo():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    estado = (0, 0b001)
    vecinos = dict(problema.vecinos(estado))
    # el estado (ciudad=1 "B", mascara=0b011) deberia costar 1 (peso A-B)
    assert vecinos[(1, 0b011)] == 1
    assert vecinos[(2, 0b101)] == 2  # peso A-C


def test_costo_cierre_ciclo_vuelve_al_inicio():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    assert problema.costo_cierre_ciclo(1) == problema.grafo.peso("B", "A")


def test_nombres_de_ruta_traduce_indices_a_nombres():
    problema = ProblemaTSP(_grafo_triangulo(), ciudad_inicio="A")
    assert problema.nombres_de_ruta([0, 1, 2]) == ["A", "B", "C"]
