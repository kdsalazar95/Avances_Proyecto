"""
Tests de grafo.py

Verifican la estructura de grafo pura: que agregar ciudades y aristas
funcione, que los pesos se consulten correctamente en ambos sentidos
(grafo no dirigido), y que los vecinos se calculen bien.
"""

import pytest
from grafo import Grafo


def test_agregar_ciudad_asigna_indices_consecutivos():
    g = Grafo()
    assert g.agregar_ciudad("A") == 0
    assert g.agregar_ciudad("B") == 1
    assert g.agregar_ciudad("A") == 0  # ya existia, no se duplica
    assert g.n == 2


def test_agregar_arista_agrega_ciudades_automaticamente():
    g = Grafo()
    g.agregar_arista("A", "B", 10)
    assert g.n == 2
    assert set(g.nodos) == {"A", "B"}


def test_peso_es_simetrico_grafo_no_dirigido():
    g = Grafo()
    g.agregar_arista("A", "B", 10)
    assert g.peso("A", "B") == 10
    assert g.peso("B", "A") == 10  # el orden no deberia importar


def test_peso_funciona_con_nombres_o_indices():
    g = Grafo()
    g.agregar_arista("A", "B", 10)
    assert g.peso("A", "B") == g.peso(0, 1)


def test_peso_de_arista_inexistente_lanza_error():
    g = Grafo()
    g.agregar_arista("A", "B", 10)
    g.agregar_ciudad("C")
    with pytest.raises(KeyError):
        g.peso("A", "C")  # nunca se agrego esa arista


def test_vecinos_de_devuelve_todas_las_conexiones():
    g = Grafo()
    g.agregar_arista("A", "B", 10)
    g.agregar_arista("A", "C", 20)
    vecinos = dict(g.vecinos_de("A"))
    assert vecinos == {g.indice_de("B"): 10, g.indice_de("C"): 20}


def test_indice_de_y_nombre_de_son_inversos():
    g = Grafo()
    g.agregar_ciudad("A")
    g.agregar_ciudad("B")
    assert g.nombre_de(g.indice_de("B")) == "B"
