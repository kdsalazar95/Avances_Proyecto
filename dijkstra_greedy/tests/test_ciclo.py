"""Pruebas del ciclo agente Greedy <-> evaluador (agente_greedy.py): una simulación de la malla,
pesos de los PEs, camino Greedy y resultados guardados, con el evaluador real."""

import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

_raiz_dg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_raiz_dg), "cgra", "src"))  # simulador (cgra/src)
sys.path.insert(0, _raiz_dg)  # Dijkstra + Greedy (dijkstra_greedy/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agente_greedy  # noqa: E402
from test_dijkstra import _ConOperaciones  # noqa: E402


class TestCicloCompleto(_ConOperaciones):
    """agente_greedy.py con el evaluador real (Dijkstra + simulador) sobre una malla 3x3."""

    def correr(self, salida, parametros, *extra):
        with mock.patch.object(agente_greedy, "operaciones_del_c", lambda: [self.pequena]), \
             mock.patch.object(agente_greedy, "CARPETA_PROGRAMAS", Path(self.programas)), \
             mock.patch.object(agente_greedy, "preparar", lambda *a, **k: []), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return agente_greedy.main(["--salida", str(salida), "--parametros", str(parametros), *extra])

    def test_camino_y_resultados(self):
        """Simula la 3x3 una vez, guarda los pesos de los 9 PEs y el camino desde PE00; no pisa resultados
        anteriores, y --continuar da el mismo camino sin volver a simular."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parametros = tmp / "agente.json"
            parametros.write_text(json.dumps(dict(malla=[3, 3], inicio=[0, 0], epsilon=0.1, semilla=0, procesos=1)))
            salida = tmp / "resultados"
            self.assertEqual(self.correr(salida, parametros), 0)
            with self.assertRaises(SystemExit):  # no pisa resultados anteriores
                self.correr(salida, parametros)
            evaluacion = json.loads((salida / "evaluacion_malla.json").read_text())
            pesos = json.loads((salida / "pesos.json").read_text())["pesos"]
            camino = json.loads((salida / "camino.json").read_text())
            resumen = json.loads((salida / "resumen.json").read_text())
            historia = json.loads((salida / "historia.json").read_text())
            with mock.patch.object(agente_greedy, "medir_en_paralelo", side_effect=AssertionError):
                self.assertEqual(self.correr(salida, parametros, "--continuar"), 0)
            otra_vez = json.loads((salida / "camino.json").read_text())
        self.assertEqual(evaluacion["evaluacion"]["estado"], "correcta")
        self.assertEqual(evaluacion["configuracion"]["malla"], ["M E M", "M V M", "M E M"])
        self.assertEqual(len(pesos), 9)
        for p in pesos:
            self.assertAlmostEqual(p["peso"], p["w_tipo"] * (1 + p["nops"] / p["ciclos"]))
        self.assertEqual(camino["inicio"], "PE00")
        pes = [p["pe"] for p in camino["camino"]]
        self.assertEqual(len(pes), len(set(pes)))
        self.assertEqual(resumen["pes_camino"], len(pes))
        self.assertEqual(resumen["malla"], "3x3")
        self.assertEqual(resumen["C_camino"], camino["valuacion"]["C"])
        self.assertEqual(set(camino["valuacion"]["por_operacion"]), {self.pequena})
        self.assertEqual(otra_vez, camino)
        self.assertEqual([p["pe"] for p in historia["pasos"]], pes)  # historia.json para la curva y el GIF
        self.assertEqual(historia["pasos"][-1]["C"], camino["valuacion"]["C"])
