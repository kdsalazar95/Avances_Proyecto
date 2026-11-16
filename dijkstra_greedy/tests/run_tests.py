"""
Corre las pruebas de Dijkstra + Greedy (dijkstra_greedy/tests/) con la misma
salida amigable que las del simulador (ResultadoAmigable de cgra/test/run_tests.py).

Cómo correrlo, desde la raíz del repositorio:
    python3 dijkstra_greedy/tests/run_tests.py
"""
import importlib.util
import os
import sys

_aqui = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _aqui)

import test_dijkstra  # noqa: E402  (también pone cgra/src y dijkstra_greedy/ en sys.path)
import test_agente  # noqa: E402
import test_ciclo  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_tests_simulador", os.path.join(os.path.dirname(os.path.dirname(_aqui)), "cgra", "test", "run_tests.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

NOMBRES_DE_SECCION = {
    "TestGrafoIR": "Bloque Dijkstra: grafo desde el IR (skill 04)",
    "TestGrafoOtrasOperaciones": "Bloque Dijkstra: grafo de otras operaciones (producto, resta, división)",
    "TestMapeoDijkstra": "Bloque Dijkstra: mapeo y ruteo (skills 05-07)",
    "TestVerificador": "Bloque Dijkstra: verificador",
    "TestVectorizacion": "Bloque Dijkstra: modo vectorial",
    "TestEvaluador": "Evaluador: costos de área, latencia y correctitud",
    "TestAgenteGreedy": "Agente Greedy: camino por los PEs de menor peso",
    "TestAgenteConEvaluador": "Agente Greedy con el evaluador real",
    "TestCompilarSkill02": "Skills 02-03 (clang, opt) dentro del ciclo",
    "TestSimuladorDijkstra": "Bloque Dijkstra: PE*.txt en el simulador",
    "TestSimuladorOtrasOperaciones": "Bloque Dijkstra: otras operaciones en el simulador",
    "TestCicloCompleto": "Ciclo agente-evaluador: simulación, pesos y camino con el evaluador real",
}


def main():
    _base.correr([
        test_dijkstra.TestGrafoIR,
        test_dijkstra.TestGrafoOtrasOperaciones,
        test_dijkstra.TestMapeoDijkstra,
        test_dijkstra.TestVerificador,
        test_dijkstra.TestVectorizacion,
        test_dijkstra.TestSimuladorDijkstra,
        test_dijkstra.TestSimuladorOtrasOperaciones,
        test_dijkstra.TestEvaluador,
        test_agente.TestAgenteGreedy,
        test_agente.TestAgenteConEvaluador,
        test_agente.TestCompilarSkill02,
        test_ciclo.TestCicloCompleto,
    ], NOMBRES_DE_SECCION)


if __name__ == "__main__":
    main()
