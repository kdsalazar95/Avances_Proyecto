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
            self.assertEqual(self.correr(salida, parametros, "--latido", "0", "--informes"), 0)
            episodios = json.loads((salida / "episodios.json").read_text())
            resumen = json.loads((salida / "resumen.json").read_text())
            convergencia = json.loads((salida / "convergencia.json").read_text())
            reporte = json.loads((salida / "reporte_corrida.json").read_text())
            texto_reporte = (salida / "reporte_corrida.md").read_text()
            tabla = (salida / "configuraciones.csv").read_text().splitlines()
            graficas = [n for n in ("frente_pareto.png", "curva_costos.png", "exploracion.gif") if (salida / n).exists()]
            cache = json.loads((salida / "evaluaciones_episodios.json").read_text())
            for archivo in ("reporte_corrida.json", "configuraciones.csv", "convergencia.json"):
                (salida / archivo).unlink()
            with mock.patch.object(agente_greedy, "medir_en_paralelo", side_effect=AssertionError):
                self.assertEqual(self.correr(salida, parametros, "--continuar", "--latido", "0"), 0)
            otra_vez = json.loads((salida / "episodios.json").read_text())
            # sin --informes la corrida deja solo los datos de convergencia (y lo que necesita --continuar)
            self.assertFalse((salida / "reporte_corrida.json").exists())
            self.assertFalse((salida / "configuraciones.csv").exists())
            convergencia_2 = json.loads((salida / "convergencia.json").read_text())
            filas = (salida / "convergencia.csv").read_text().splitlines()
        # datos de convergencia: la referencia es n = 0, cada malla propuesta una fila, tiempos reales
        self.assertEqual(convergencia["sistema"], "sistema2_greedy")
        self.assertEqual(convergencia["evaluaciones"], len(filas) - 2)  # menos el encabezado y la referencia
        self.assertGreater(convergencia["tiempo_total_s"], 0)  # Dijkstra + simulador de verdad
        self.assertEqual(convergencia["evaluaciones_sin_tiempo"], 0)
        # con --continuar todo sale de la caché, pero los tiempos son los de cuando se simuló
        def sin_tiempos(c):
            otros = ("tiempo_total_s", "convergencia", "hitos", "por_paciencia")
            return {**{k: v for k, v in c.items() if k not in otros},
                    "convergencia": c["convergencia"]["evaluacion"],
                    "hitos": {p: (h or {}).get("evaluacion") for p, h in c["hitos"].items()},
                    "por_paciencia": {p: sin_tiempos(r) for p, r in (c.get("por_paciencia") or {}).items()}}
        self.assertEqual(sin_tiempos(convergencia_2), sin_tiempos(convergencia))
        self.assertAlmostEqual(convergencia_2["tiempo_total_s"], convergencia["tiempo_total_s"], delta=0.05)
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
            # (y el tiempo de decisión de Greedy, que se mide en cada corrida)
            return [{**{k: v for k, v in e.items() if k not in ("nuevas", "duracion_s")},
                     "trayectoria": [{k: v for k, v in t.items() if k not in ("nueva", "tiempo_decision_s")}
                                     for t in e["trayectoria"]]}
                    for e in eps]
        self.assertEqual(sin_contadores(otra_vez["episodios"]), sin_contadores(episodios["episodios"]))
        self.assertEqual([e["nuevas"] for e in otra_vez["episodios"]], [0] * len(otra_vez["episodios"]))

    def test_barrido_de_epsilon(self):
        """herramientas/barrido.py: una corrida por epsilon con el evaluador real; la referencia se simula una
        sola vez y cada corrida hereda las simulaciones de la anterior; la comparación junta las corridas en el
        formato del Sistema 1; volver a correr el barrido no repite corridas terminadas."""
        sys.path.insert(0, os.path.join(_raiz_dg, "herramientas"))
        import barrido
        import convergencia
        self.assertEqual(barrido.leer_epsilons(["0", "10", "20", "80", "90", "100"]), [0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parametros = tmp / "agente.json"
            parametros.write_text(json.dumps(dict(malla=[3, 3], inicio=[0, 0], epsilon=0.0, semilla=0, procesos=1,
                                                  modo="bloques", bloques=[1], max_episodios=2, paciencia=1)))
            medidas = []
            real = agente_greedy.medir_en_paralelo

            def contar(conf, *a, **k):
                medidas.append(tuple(conf["malla"]))
                return real(conf, *a, **k)

            def correr(argv):
                salida = Path(argv[argv.index("--salida") + 1])
                return self.correr(salida, parametros, *[x for x in argv if x != str(salida) and x != "--salida"],
                                   "--latido", "0")

            with mock.patch.object(agente_greedy, "medir_en_paralelo", contar):
                carpetas, resumenes = barrido.barrido([0.0, 1.0], "s2", tmp / "resultados", correr=correr)
            self.assertEqual([c.name for c in carpetas], ["s2-eps000", "s2-eps100"])
            self.assertEqual(medidas.count(("M E M", "M V M", "M E M")), 1)  # la referencia, una vez
            self.assertEqual(len(medidas), len(set(medidas)))  # ninguna malla se simula dos veces
            # una corrida por epsilon con paciencia 10 y tope 30; la comparación tiene cada epsilon x paciencia
            self.assertEqual([(r["epsilon"], r["paciencia"]) for r in resumenes],
                             [(0.0, 3), (0.0, 6), (0.0, 10), (1.0, 3), (1.0, 6), (1.0, 10)])
            episodios_1 = json.loads((carpetas[1] / "episodios.json").read_text())
            self.assertEqual((episodios_1["epsilon"], episodios_1["paciencia"], episodios_1["max_episodios"]),
                             (1.0, 10, 30))
            tabla = (tmp / "resultados" / "s2-comparacion" / "comparacion_epsilon.csv").read_text().splitlines()
            self.assertEqual(tabla[0].split(","), list(convergencia.COLUMNAS_COMPARACION))
            self.assertEqual(len(tabla), 7)
            self.assertEqual(resumenes[-1]["fraccion_exploracion"], 1.0)
            episodios = [r["episodios"] for r in resumenes[3:]]
            self.assertEqual(episodios, sorted(episodios))  # más paciencia, al menos tantos episodios
            # cortar la corrida de paciencia 10 da lo mismo que correr con paciencia 3
            directa = tmp / "directa"
            self.assertEqual(self.correr(directa, parametros, "--epsilon", "1.0", "--paciencia", "3",
                                         "--max-episodios", "30", "--latido", "0"), 0)
            n, motivo = convergencia.corte(episodios_1["episodios"], 3, episodios_1["motivo_fin"])
            real = json.loads((directa / "episodios.json").read_text())
            self.assertEqual((n, motivo), (len(real["episodios"]), real["motivo_fin"]))

            def sin_tiempos(filas):
                return [{k: v for k, v in f.items() if not k.startswith("tiempo") and k not in ("corrida",)}
                        for f in filas]
            self.assertEqual(sin_tiempos(convergencia.filas_corrida(carpetas[1], n)),
                             sin_tiempos(convergencia.filas_corrida(directa)))
            # ya terminadas: no se vuelven a correr
            self.assertEqual(barrido.barrido([0.0, 1.0], "s2", tmp / "resultados",
                                             correr=lambda argv: self.fail("no debía correr"))[0], carpetas)

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
