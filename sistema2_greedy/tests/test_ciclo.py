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
sys.path.insert(0, _raiz_dg)  # Dijkstra + Greedy (sistema2_greedy/)
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
        """Simula la 3x3 una vez, guarda los pesos de los 9 PEs y el camino desde PE00; corre el ciclo de
        episodios (cada rectángulo simulado una vez) y guarda el C real, la recompensa y el mejor; no pisa
        resultados anteriores, y --continuar da los mismos episodios sin volver a simular."""
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
            episodios = json.loads((salida / "episodios.json").read_text())
            cache = json.loads((salida / "evaluaciones_episodios.json").read_text())
            with mock.patch.object(agente_greedy, "medir_en_paralelo", side_effect=AssertionError):
                self.assertEqual(self.correr(salida, parametros, "--continuar"), 0)
            otra_vez = json.loads((salida / "camino.json").read_text())
            episodios_otra_vez = json.loads((salida / "episodios.json").read_text())
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
        def sin_duracion(c):  # la duración del episodio cambia de una corrida a otra
            return {**c, "episodio": {k: v for k, v in c["episodio"].items() if k != "duracion_s"}}
        self.assertEqual(sin_duracion(otra_vez), sin_duracion(camino))
        self.assertEqual([p["pe"] for p in historia["pasos"]], pes)  # historia.json para la curva y el GIF
        self.assertEqual(historia["pasos"][-1]["C"], camino["valuacion"]["C"])
        # ciclo de episodios: cada rectángulo simulado una vez; camino.json es el del mejor episodio
        lista = episodios["episodios"]
        self.assertEqual(len(lista), resumen["episodios"])
        self.assertIn(episodios["motivo_fin"], ("paciencia", "max_episodios"))
        self.assertEqual([ep["semilla"] for ep in lista], list(range(len(lista))))
        mallas = {tuple(ep["malla"]) for ep in lista}
        self.assertEqual({tuple(e["configuracion"]["malla"]) for e in cache["episodios"]}, mallas)
        self.assertEqual(len(cache["episodios"]), len(mallas))  # sin simulaciones repetidas
        self.assertEqual(camino["episodio"]["n"], resumen["episodio_mejor"] if resumen["episodio_mejor"] is not None
                         else lista[-1]["n"])
        self.assertEqual(resumen["malla_episodio"], camino["episodio"]["malla"])
        self.assertEqual([{k: v for k, v in e.items() if k != "duracion_s"} for e in episodios_otra_vez["episodios"]],
                         [{k: v for k, v in e.items() if k != "duracion_s"} for e in lista])

    def test_modo_bloques(self):
        """Modo bloques con el evaluador real: parte de la 3x3 (simulada una vez), prueba quitar una fila,
        guarda episodios con su descenso, la caché y el resumen; --continuar no vuelve a simular."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parametros = tmp / "agente.json"
            parametros.write_text(json.dumps(dict(malla=[3, 3], inicio=[0, 0], epsilon=0.0, semilla=0, procesos=1,
                                                  modo="bloques", bloques=[1], max_episodios=2, paciencia=1)))
            salida = tmp / "resultados"
            self.assertEqual(self.correr(salida, parametros, "--latido", "0"), 0)
            episodios = json.loads((salida / "episodios.json").read_text())
            resumen = json.loads((salida / "resumen.json").read_text())
            reporte = json.loads((salida / "reporte_corrida.json").read_text())
            texto_reporte = (salida / "reporte_corrida.md").read_text()
            tabla = (salida / "configuraciones.csv").read_text().splitlines()
            graficas = [n for n in ("frente_pareto.png", "curva_costos.png", "exploracion.gif") if (salida / n).exists()]
            cache = json.loads((salida / "evaluaciones_episodios.json").read_text())
            with mock.patch.object(agente_greedy, "medir_en_paralelo", side_effect=AssertionError):
                self.assertEqual(self.correr(salida, parametros, "--continuar", "--latido", "0"), 0)
            otra_vez = json.loads((salida / "episodios.json").read_text())
        self.assertEqual(episodios["modo"], "bloques")
        # reporte, tabla y gráficas al final de la corrida
        self.assertEqual(reporte["configuraciones"]["evaluadas"], len(tabla) - 1)
        dj = reporte["dijkstra_y_simulacion"]
        self.assertGreater(dj["mapeos"], 0)  # las mallas nuevas registran el tiempo de Dijkstra
        self.assertGreater(dj["tiempo_mapeo_total_s"], 0)
        self.assertGreater(reporte["recursos"]["cpu_total_s"], 0)
        self.assertIn("## Recursos computacionales", texto_reporte)
        self.assertEqual(graficas, ["frente_pareto.png", "curva_costos.png", "exploracion.gif"])
        self.assertEqual(episodios["malla_referencia"], ["M E M", "M V M", "M E M"])
        ep = episodios["episodios"][0]
        self.assertEqual(ep["trayectoria"][0]["accion"], "inicio")
        self.assertTrue(all(t["accion"] in ("inicio", "FILAS INICIO", "FILAS FIN") for t in ep["trayectoria"]))
        self.assertGreaterEqual(ep["simuladas"], 1)
        mallas = {tuple(e["configuracion"]["malla"]) for e in cache["episodios"]}
        self.assertIn(("M E M", "M V M", "M E M"), mallas)
        self.assertEqual(resumen["modo"], "bloques")
        self.assertEqual(resumen["configuraciones_simuladas"], len(cache["episodios"]))
        def sin_contadores(eps):  # con --continuar todo sale de la caché: cambian solo "nuevas"/"nueva"
            return [{**{k: v for k, v in e.items() if k not in ("nuevas", "duracion_s")},
                     "trayectoria": [{k: v for k, v in t.items() if k != "nueva"} for t in e["trayectoria"]]}
                    for e in eps]
        self.assertEqual(sin_contadores(otra_vez["episodios"]), sin_contadores(episodios["episodios"]))
        self.assertEqual([e["nuevas"] for e in otra_vez["episodios"]], [0] * len(otra_vez["episodios"]))

    def test_continuar_no_reusa_con_otro_programa(self):
        """Si cambió operaciones.c (o costos/normalizadores), --continuar no reusa la simulación guardada."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parametros = tmp / "agente.json"
            parametros.write_text(json.dumps(dict(malla=[3, 3], inicio=[0, 0], epsilon=0.0, semilla=0, procesos=1,
                                                  modo="bloques", bloques=[1], max_episodios=1, paciencia=1)))
            salida = tmp / "resultados"
            self.assertEqual(self.correr(salida, parametros, "--latido", "0"), 0)
            guardada = json.loads((salida / "evaluacion_malla.json").read_text())
            self.assertIn("huella", guardada)
            with mock.patch.object(agente_greedy, "huella_entradas", return_value="otra"), \
                 mock.patch.object(agente_greedy, "medir_en_paralelo", side_effect=AssertionError("re-simula")):
                with self.assertRaisesRegex(AssertionError, "re-simula"):
                    self.correr(salida, parametros, "--continuar", "--latido", "0")
