"""Pruebas del agente Greedy (dijkstra_greedy/dijkstra/agente.py): camino sin ciclos por los PEs
de una malla, siempre al vecino sin visitar que deja el menor costo C del camino (P_fallo si no
cumple los tipos de PE), hasta que cumple los tipos y ningún vecino baja C."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_raiz_dg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_raiz_dg), "cgra", "src"))  # simulador (cgra/src)
sys.path.insert(0, _raiz_dg)  # Dijkstra + Greedy (dijkstra_greedy/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_dg, "herramientas"))  # gif_camino.py

from dijkstra.agente import Agente, Parametros, PesoPE, medir_en_paralelo
from dijkstra.rl_llm import (EleccionLLM, construir_prompt, construir_prompt_parada, interpretar,
                             interpretar_parada)
from dijkstra.compilar import FUENTE, ErrorCompilacion, grafos, preparar
from dijkstra.evaluador import RUTA_COSTOS, Evaluacion, EvaluacionOperacion, Evaluador
from formato_cgra import RUTA_POR_DEFECTO
from malla import Malla
from run_cgra import SimuladorCGRA
from test_dijkstra import _ConOperaciones

with open(RUTA_POR_DEFECTO, encoding="utf-8") as _archivo:
    BASE = json.load(_archivo)

W = {"M": 12, "E": 1, "V": 2}  # w_tipo con los pesos w1-w4 de costos.json
MALLA_3X3 = (("M", "E", "M"), ("M", "V", "M"), ("M", "E", "M"))


def evaluacion_falsa(malla, nops_por_op, ciclos=10, estado="correcta"):
    """Una Evaluacion como la del evaluador, con ``nops_por_pe`` por operación."""
    operaciones = {op: EvaluacionOperacion(op, estado, ciclos=ciclos, nops=sum((n or {}).values()), nops_por_pe=n,
                                           C_R=0.0)
                   for op, n in nops_por_op.items()}
    normalizadores = {"C_A_ref": 76, "operaciones": {op: {"C_A_ref": 76, "C_L_ref": 100.0} for op in nops_por_op}}
    return Evaluacion([" ".join(f) for f in malla], estado, 1.0, {}, "", operaciones, normalizadores=normalizadores)


def nops_iguales(malla, valor):
    filas, columnas = len(malla), len(malla[0])
    return {Malla(filas, columnas).id_de(f, c): valor for f in range(filas) for c in range(columnas)}


def parametros(**cambios):
    datos = dict(malla=[3, 3], inicio=[0, 0], epsilon=0.0, semilla=0, procesos=1)  # epsilon 0: Greedy puro
    datos.update(cambios)
    return Parametros(**datos)


class TestAgenteGreedy(unittest.TestCase):
    """Greedy camina por los PEs: siempre al vecino sin visitar de menor costo C, sin repetir ninguno."""

    def test_1_peso_de_cada_pe(self):
        """Peso = w_tipo · (1 + NOPs/ciclos), con NOPs y ciclos sumados sobre las operaciones."""
        agente = Agente(None, BASE, parametros())
        nops = {"op1": nops_iguales(MALLA_3X3, 2), "op2": nops_iguales(MALLA_3X3, 3)}
        nops["op1"]["PE11"] = 10  # el V de en medio está en NOP los 10 ciclos de op1
        pesos = agente.calcular_pesos(evaluacion_falsa(MALLA_3X3, nops, ciclos=10))
        self.assertEqual(pesos["PE00"].w_tipo, 12)
        self.assertEqual(pesos["PE00"].ciclos, 20)
        self.assertAlmostEqual(pesos["PE00"].peso, 12 * (1 + 5 / 20))
        self.assertAlmostEqual(pesos["PE01"].peso, 1 * (1 + 5 / 20))
        self.assertAlmostEqual(pesos["PE11"].peso, 2 * (1 + 13 / 20))
        self.assertEqual({p.tipo for p in pesos.values()}, {"M", "E", "V"})

    def test_2_sin_reporte_por_pe_falla(self):
        """Sin la tabla de NOPs por PE del reporte de ciclos no hay pesos."""
        agente = Agente(None, BASE, parametros())
        evaluacion = evaluacion_falsa(MALLA_3X3, {"op1": None})
        with self.assertRaisesRegex(ValueError, "sin reporte de ciclos por PE"):
            agente.calcular_pesos(evaluacion)

    def test_3_aristas_son_los_vecinos(self):
        """Cada PE se conecta con sus vecinos norte, sur, este y oeste dentro de la malla."""
        self.assertEqual(sorted(Agente.vecinos("PE00", 3, 3)), ["PE01", "PE10"])
        self.assertEqual(sorted(Agente.vecinos("PE01", 3, 3)), ["PE00", "PE02", "PE11"])
        self.assertEqual(sorted(Agente.vecinos("PE11", 3, 3)), ["PE01", "PE10", "PE12", "PE21"])
        self.assertEqual(sorted(Agente.vecinos("PE0000", 12, 12)), ["PE0001", "PE0100"])

    def test_4_camino_greedy_sin_ciclos(self):
        """Empieza en PE00; cada paso va al vecino sin visitar que deja el menor C efectivo; ningún PE
        se repite y, al terminar por ``sin_mejora``, el camino cumple los tipos y ningún vecino baja C."""
        malla = Agente(None, BASE, parametros(malla=[4, 5])).malla()
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(4, 5).ids())}}
        agente = Agente(lambda c: evaluacion_falsa(malla, nops), BASE, parametros(malla=[4, 5]))
        camino = agente.buscar()
        self.assertEqual(camino[0], "PE00")
        self.assertEqual(len(camino), len(set(camino)))
        for i, (actual, siguiente) in enumerate(zip(camino, camino[1:])):
            libres = [v for v in Agente.vecinos(actual, 4, 5) if v not in camino[:i + 1]]
            self.assertIn(siguiente, libres)
            prefijo = camino[:i + 1]
            self.assertEqual(agente.costo_efectivo(prefijo + [siguiente]),
                             min(agente.costo_efectivo(prefijo + [v]) for v in libres))
        libres = [v for v in Agente.vecinos(camino[-1], 4, 5) if v not in camino]
        self.assertIn(agente.parada, ("sin_mejora", "sin_vecinos"))
        if agente.parada == "sin_vecinos":
            self.assertEqual(libres, [])
        else:
            self.assertTrue(agente.cumple_tipos({agente.pesos[pe].tipo for pe in camino}))
            c = agente.costo_efectivo(camino)[0]
            self.assertTrue(all(agente.costo_efectivo(camino + [v])[0] >= c for v in libres))

    def test_5_empates_y_paro_sin_mejora(self):
        """Sin NOPs: PE00 (M) y sus vecinos PE01 (E) y PE10 (M) no cumplen los tipos (los dos cuestan
        P_fallo); desempata el C valuado, más bajo con E. Luego PE11 (V) cumple los tipos. Desde ahí
        cualquier PE solo suma área, C sube, y el camino termina (sin_mejora)."""
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)}), BASE,
                        parametros())
        camino = agente.buscar()
        self.assertEqual(camino, ["PE00", "PE01", "PE11"])
        self.assertEqual(agente.parada, "sin_mejora")
        self.assertEqual(camino, Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)}),
                                        BASE, parametros()).buscar())

    def test_6_inicio_configurable(self):
        """El camino empieza en el PE de ``inicio``; fuera de la malla es un error."""
        evaluacion = evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)})
        self.assertEqual(Agente(lambda c: evaluacion, BASE, parametros(inicio=[2, 2])).buscar()[0], "PE22")
        with self.assertRaisesRegex(ValueError, "fuera de la malla"):
            Agente(lambda c: evaluacion, BASE, parametros(inicio=[3, 0])).buscar()

    def test_7_una_sola_simulacion(self):
        """La malla se evalúa una sola vez, con la configuración de cgra.json en ese tamaño."""
        llamadas = []

        def medir(configuracion):
            llamadas.append(configuracion)
            return evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 1)})

        agente = Agente(medir, BASE, parametros())
        agente.buscar()
        self.assertEqual(len(llamadas), 1)
        self.assertEqual(llamadas[0]["malla"], ["M E M", "M V M", "M E M"])
        self.assertNotIn("distribucion", llamadas[0])
        self.assertEqual(llamadas[0]["tipos_pe"], BASE["tipos_pe"])

    def test_8_malla_incorrecta_no_camina(self):
        """Si la simulación de la malla no es correcta, no hay pesos ni camino."""
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)},
                                                   estado="incorrecta"), BASE, parametros())
        with self.assertRaisesRegex(ValueError, "no es correcta"):
            agente.buscar()

    def test_9_resultado(self):
        """El resultado son los PEs del camino (no tienen que formar un rectángulo), con su peso total,
        cuántos de cada tipo y el mapa con '.' en los que quedan fuera."""
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)}), BASE,
                        parametros())
        camino = agente.buscar()
        r = agente.resultado()
        self.assertEqual([p["pe"] for p in r["camino"]], camino)
        self.assertEqual(r["pes"], len(camino))
        self.assertAlmostEqual(r["peso_total"], sum(agente.pesos[pe].peso for pe in camino))
        self.assertEqual(sum(r["por_tipo"].values()), len(camino))
        letras = [t for fila in r["mapa"] for t in fila.split()]
        self.assertEqual(sum(t != "." for t in letras), len(camino))
        self.assertEqual(r["inicio"], "PE00")


    def test_10_valuacion_del_camino(self):
        """El camino se valúa con la función de costo sin volver a simular: C_A = Σ w_tipo del camino,
        C_L = ciclos·(1 + NOPs del camino/(ciclos·PEs del camino)), C = K_A·C_A/C_A_ref + K_L·C_L/C_L_ref + K_R·C_R."""
        nops = nops_iguales(MALLA_3X3, 4)
        llamadas = []
        agente = Agente(lambda c: llamadas.append(c) or evaluacion_falsa(MALLA_3X3, {"op": nops}, ciclos=10),
                        BASE, parametros())
        camino = agente.buscar()
        v = agente.valuar_camino()
        self.assertEqual(len(llamadas), 1)  # la única simulación es la de la malla
        c_a = sum(W[agente.pesos[pe].tipo] for pe in camino)
        c_l = 10 * (1 + 4 * len(camino) / (10 * len(camino)))
        self.assertEqual(v["C_A"], c_a)
        self.assertAlmostEqual(v["por_operacion"]["op"]["C_L"], c_l)
        self.assertAlmostEqual(v["C"], c_a / 76 + c_l / 100.0 + 0.0)
        self.assertEqual(agente.resultado()["valuacion"], v)


    def test_11_epsilon_greedy(self):
        """Con epsilon, cada paso explora (un vecino libre al azar) o explota (el de menor C); el camino
        sigue sin repetir PEs, y la misma semilla da el mismo camino."""
        malla = Agente(None, BASE, parametros(malla=[5, 6])).malla()
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(5, 6).ids())}}

        def correr(**cambios):
            agente = Agente(lambda c: evaluacion_falsa(malla, nops), BASE, parametros(malla=[5, 6], **cambios))
            return agente, agente.buscar()

        greedy, camino_greedy = correr(epsilon=0.0)
        self.assertEqual(set(greedy.decisiones[1:]), {"explotacion"})
        agente, camino = correr(epsilon=0.5, semilla=3)
        self.assertEqual(camino, correr(epsilon=0.5, semilla=3)[1])  # misma semilla, mismo camino
        self.assertEqual(len(camino), len(set(camino)))
        self.assertEqual(agente.decisiones[0], "inicio")
        self.assertEqual(len(agente.decisiones), len(camino))
        self.assertIn("exploracion", agente.decisiones)
        for i, (actual, siguiente) in enumerate(zip(camino, camino[1:])):
            libres = [v for v in Agente.vecinos(actual, 5, 6) if v not in camino[:i + 1]]
            self.assertIn(siguiente, libres)
            if agente.decisiones[i + 1] == "explotacion":
                prefijo = camino[:i + 1]
                self.assertEqual(agente.costo_efectivo(prefijo + [siguiente]),
                                 min(agente.costo_efectivo(prefijo + [v]) for v in libres))
        todo_azar, _ = correr(epsilon=1.0, semilla=3)
        self.assertEqual(set(todo_azar.decisiones[1:]), {"exploracion"})
        r = agente.resultado()
        self.assertEqual(r["exploraciones"], agente.decisiones.count("exploracion"))
        self.assertEqual([p["decision"] for p in r["camino"]], agente.decisiones)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            correr(epsilon=1.5)

    def test_12_historia_del_camino(self):
        """La historia valúa cada prefijo del camino (el último es la valuación del camino), dice si cumple
        los tipos de PE (memoria, E y V) y marca el de menor C entre los que cumplen."""
        nops = nops_iguales(MALLA_3X3, 4)
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops}), BASE, parametros())
        camino = agente.buscar()
        h = agente.historia()
        self.assertEqual([p["pe"] for p in h["pasos"]], camino)
        self.assertEqual([p["decision"] for p in h["pasos"]], agente.decisiones)
        self.assertEqual([p["pes"] for p in h["pasos"]], list(range(1, len(camino) + 1)))
        for i, p in enumerate(h["pasos"]):
            self.assertAlmostEqual(p["C"], agente.valuar_camino(camino[:i + 1])["C"])
            self.assertEqual((p["fila"], p["columna"]), Malla.posicion(p["pe"]))
            tipos = {agente.pesos[pe].tipo for pe in camino[:i + 1]}
            self.assertEqual(p["cumple_tipos"], {"M", "E", "V"} <= tipos)
        self.assertAlmostEqual(h["pasos"][-1]["C"], agente.valuar_camino()["C"])
        validos = [p for p in h["pasos"] if p["cumple_tipos"]]
        self.assertEqual(h["mejor"]["C"], min(p["C"] for p in validos))
        self.assertEqual(h["malla"], [" ".join(f) for f in MALLA_3X3])
        self.assertEqual((h["filas"], h["columnas"]), (3, 3))
        json.dumps(h, allow_nan=False)  # se puede guardar como historia.json

    def test_13_curva_y_gif_deterministicos(self):
        """herramientas/gif_camino.py dibuja la curva y el GIF desde historia.json, igual byte por byte."""
        import hashlib
        import gif_camino
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(3, 3).ids())}}
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, nops), BASE, parametros(epsilon=0.5, semilla=1))
        agente.buscar()
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp)
            (corrida / "historia.json").write_text(json.dumps(agente.historia()), encoding="utf-8")
            resumen = []
            for salida in ("a", "b"):
                with mock.patch("sys.stdout"):
                    self.assertEqual(gif_camino.main([str(corrida), "--salida", str(corrida / salida)]), 0)
                resumen.append([hashlib.sha256((corrida / salida / n).read_bytes()).hexdigest()
                                for n in ("curva_costo.png", "camino.gif")])
            self.assertEqual(resumen[0], resumen[1])
            from PIL import Image
            with Image.open(corrida / "a" / "camino.gif") as g:
                self.assertEqual(g.n_frames, len(agente.camino))
        self.assertEqual(gif_camino._cuadros(10, 4), [0, 3, 6, 9])
        self.assertEqual(gif_camino._cuadros(10, 3), [0, 4, 8, 9])


    def test_14_sigue_mientras_baja_el_costo(self):
        """El paro es por costo, no por tipos: si agregar un PE baja C (aquí, un E sin NOPs diluye los
        NOPs del camino en C_L más de lo que suma de área), el camino sigue; para cuando todo vecino
        sube C. La recompensa de cada paso es C_efectivo anterior − C_efectivo."""
        nops = nops_iguales(MALLA_3X3, 0)
        nops.update(PE00=10, PE01=10, PE11=10)
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops}, ciclos=10), BASE, parametros())
        camino = agente.buscar()
        # Tras [PE00, PE01, PE11] (cumple tipos): PE21 (E, sin NOPs) baja C; PE10 y PE12 (M) lo suben.
        self.assertEqual(camino, ["PE00", "PE01", "PE11", "PE21"])
        self.assertEqual(agente.parada, "sin_mejora")
        h = agente.historia()
        self.assertEqual(h["parada"], "sin_mejora")
        efectivos = [agente.costo_efectivo(camino[:i + 1])[0] for i in range(len(camino))]
        self.assertEqual([p["C_efectivo"] for p in h["pasos"]], efectivos)
        self.assertIsNone(h["pasos"][0]["recompensa"])
        for i in range(1, len(camino)):
            self.assertAlmostEqual(h["pasos"][i]["recompensa"], efectivos[i - 1] - efectivos[i])
        self.assertEqual(efectivos[:2], [agente.costos.p_fallo] * 2)  # sin V no cumple los tipos
        self.assertGreater(h["pasos"][3]["recompensa"], 0)  # PE21 bajó el costo
        self.assertEqual(agente.resultado()["parada"], "sin_mejora")

    def test_15_paro_sin_vecinos(self):
        """Si se acaban los vecinos antes de cumplir los tipos o mientras C baja, para por sin_vecinos."""
        malla = (("M", "E", "V"),)
        agente = Agente(lambda c: evaluacion_falsa(malla, {"op": nops_iguales(malla, 0)}), BASE,
                        parametros(malla=["M E V"]))
        self.assertEqual(agente.buscar(), ["PE00", "PE01", "PE02"])
        self.assertEqual(agente.parada, "sin_vecinos")


class TestEpisodio(unittest.TestCase):
    """El episodio: el camino se convierte en el rectángulo que lo contiene (memoria al oeste y al este),
    se simula una vez y su C real da la recompensa R = C_best_prev − C (C_best_prev = C de la malla)."""

    FILAS = ["M E E M", "M V V M", "M E E M"]
    MALLA = tuple(tuple(f.split()) for f in FILAS)

    def agente(self, medir=None):
        medir = medir or (lambda c: evaluacion_falsa(self.MALLA, {"op": nops_iguales(self.MALLA, 0)}))
        return Agente(medir, BASE, parametros(malla=self.FILAS))

    def test_1_rectangulo_del_camino(self):
        """Filas y columnas que ocupa el camino; las columnas solo de memoria se quitan y se pone una
        columna M al oeste y otra al este. Los PEs del rectángulo fuera del camino conservan su tipo."""
        agente = self.agente()
        self.assertEqual(agente.letra_memoria(), "M")
        self.assertEqual(agente.malla_episodio(["PE00", "PE01", "PE11"]), (("M", "E", "M"), ("M", "V", "M")))
        self.assertEqual(agente.malla_episodio(["PE01", "PE02", "PE12"]),
                         (("M", "E", "E", "M"), ("M", "V", "V", "M")))
        self.assertEqual(agente.malla_episodio(["PE02", "PE03"]), (("M", "E", "M"),))
        self.assertEqual(agente.malla_episodio(["PE01", "PE11", "PE21"]),
                         (("M", "E", "M"), ("M", "V", "M"), ("M", "E", "M")))
        with self.assertRaisesRegex(ValueError, "no hay camino"):
            agente.malla_episodio([])

    def test_2_simula_el_rectangulo_y_da_la_recompensa(self):
        """La segunda simulación es la del rectángulo del camino; R = C de la malla − C del episodio."""
        llamadas = []

        def medir(configuracion):
            llamadas.append(configuracion)
            e = evaluacion_falsa(self.MALLA, {"op": nops_iguales(self.MALLA, 0)})
            e.costo = 5.0 if len(llamadas) == 1 else 1.25  # malla completa, luego episodio
            return e

        agente = self.agente(medir)
        camino = agente.buscar()
        self.assertEqual(camino, ["PE00", "PE01", "PE11"])
        ep = agente.evaluar_episodio()
        self.assertEqual(len(llamadas), 2)
        self.assertEqual(llamadas[1]["malla"], ["M E M", "M V M"])
        self.assertEqual((llamadas[1]["filas"], llamadas[1]["columnas"]), (2, 3))
        self.assertEqual(llamadas[1]["tipos_pe"], BASE["tipos_pe"])
        self.assertEqual(ep["malla"], ["M E M", "M V M"])
        self.assertEqual(ep["estado"], "correcta")
        self.assertEqual(ep["C"], 1.25)
        self.assertEqual(ep["C_best_prev"], 5.0)
        self.assertAlmostEqual(ep["recompensa"], 5.0 - 1.25)
        self.assertAlmostEqual(ep["C_estimado"], agente.valuar_camino()["C"])
        self.assertEqual(agente.resultado()["episodio"], ep)
        json.dumps(ep, allow_nan=False)

    def test_3_episodio_penalizado(self):
        """Si la simulación del rectángulo no es correcta, la recompensa es −P_fallo."""
        agente = self.agente()
        agente.buscar()
        malo = evaluacion_falsa(self.MALLA, {"op": nops_iguales(self.MALLA, 0)}, estado="incorrecta")
        malo.costo = agente.costos.p_fallo
        ep = agente.evaluar_episodio(malo)
        self.assertEqual(ep["estado"], "incorrecta")
        self.assertEqual(ep["recompensa"], -agente.costos.p_fallo)

    def test_4_sin_camino_no_hay_episodio(self):
        with self.assertRaisesRegex(ValueError, "primero hay que recorrer"):
            self.agente().evaluar_episodio()


class TestCicloEpisodios(unittest.TestCase):
    """La retroalimentación: episodios con semilla + n; cada rectángulo se simula (una vez) y su C real
    decide el mejor (pág. 5: si baja, es el mejor; si no, se descarta), la paciencia y el historial del RL."""

    def malla_y_nops(self, filas=4, columnas=5):
        malla = Agente(None, BASE, parametros(malla=[filas, columnas])).malla()
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(filas, columnas).ids())}}
        return malla, nops

    def medidor(self, malla, nops, costo_rect, estado_rect="correcta"):
        """La primera llamada es la malla completa (C = 5); las siguientes, rectángulos con
        ``costo_rect(configuracion)``. Guarda las configuraciones en ``self.llamadas``."""
        self.llamadas = []

        def medir(configuracion):
            self.llamadas.append(configuracion)
            e = evaluacion_falsa(malla, nops, estado="correcta" if len(self.llamadas) == 1 else estado_rect)
            e.costo = 5.0 if len(self.llamadas) == 1 else costo_rect(configuracion)
            return e
        return medir

    def test_1_paciencia_y_rectangulos_repetidos(self):
        """Greedy puro da siempre el mismo camino: el episodio 0 mejora y los siguientes repiten el
        rectángulo (no se vuelve a simular), no mejoran y agotan la paciencia."""
        malla, nops = self.malla_y_nops()
        medir = self.medidor(malla, nops, lambda c: 1.5)
        agente = Agente(medir, BASE, parametros(malla=[4, 5], paciencia=3, max_episodios=10))
        historial = agente.correr()
        self.assertEqual(len(historial), 4)
        self.assertEqual(agente.motivo_fin, "paciencia")
        self.assertEqual(len(self.llamadas), 2)  # la malla y un solo rectángulo
        self.assertEqual([ep["repetido"] for ep in historial], [False, True, True, True])
        self.assertEqual([ep["mejora"] for ep in historial], [True, False, False, False])
        self.assertEqual([ep["semilla"] for ep in historial], [0, 1, 2, 3])
        self.assertEqual(historial[0]["C_best_prev"], 5.0)
        self.assertAlmostEqual(historial[0]["recompensa"], 5.0 - 1.5)
        self.assertEqual(historial[1]["C_best_prev"], 1.5)
        self.assertEqual(historial[1]["recompensa"], 0.0)
        self.assertEqual(agente.c_best, 1.5)
        self.assertIs(agente.mejor, historial[0])
        self.assertEqual(agente.episodio, historial[0])

    def test_2_max_episodios(self):
        malla, nops = self.malla_y_nops()
        agente = Agente(self.medidor(malla, nops, lambda c: 1.5), BASE,
                        parametros(malla=[4, 5], paciencia=50, max_episodios=2))
        self.assertEqual(len(agente.correr()), 2)
        self.assertEqual(agente.motivo_fin, "max_episodios")
        with self.assertRaisesRegex(ValueError, "al menos 1"):
            Agente(self.medidor(malla, nops, lambda c: 1.5), BASE,
                   parametros(malla=[4, 5], max_episodios=0)).correr()

    def test_3_el_mejor_y_la_recompensa_contra_c_best(self):
        """Con exploración, los rectángulos cambian. C_best_prev es el mejor C real anterior (al
        principio, el de la malla); R = C_best_prev − C; solo mejora si baja C_best; al final el camino
        del agente es el del mejor episodio."""
        malla, nops = self.malla_y_nops()
        costo = lambda c: 0.1 * c["filas"] * c["columnas"] + 0.01 * sum(f.count("V") for f in c["malla"])
        agente = Agente(self.medidor(malla, nops, costo), BASE,
                        parametros(malla=[4, 5], epsilon=1.0, semilla=7, paciencia=4, max_episodios=8))
        historial = agente.correr()
        c_best = 5.0
        for ep in historial:
            self.assertEqual(ep["C_best_prev"], c_best)
            self.assertAlmostEqual(ep["recompensa"], c_best - ep["C"])
            self.assertEqual(ep["mejora"], ep["C"] < c_best)
            c_best = min(c_best, ep["C"])
            self.assertEqual(ep["C_best"], c_best)
        self.assertEqual(agente.c_best, min(ep["C"] for ep in historial))
        self.assertEqual(agente.camino, agente.mejor["camino"])
        self.assertEqual(agente.decisiones, agente.mejor["decisiones"])
        self.assertEqual(agente.resultado()["episodio"]["n"], agente.mejor["n"])
        self.assertEqual(agente.historia()["semilla"], agente.mejor["semilla"])
        json.dumps(historial, allow_nan=False)

    def test_4_el_rl_recibe_el_historial_con_el_c_real(self):
        """En cada paso RL, el contexto trae los últimos historial_llm episodios (con su C real) y el mejor."""
        malla, nops = self.malla_y_nops()
        vistos = []

        def rl(actual, libres, pesos, camino, opciones, c_actual, contexto, parada=None):
            if parada:
                return 0
            vistos.append((contexto["episodio"], [ep["n"] for ep in contexto["historial"]],
                           None if contexto["mejor"] is None else contexto["mejor"]["n"], contexto["C_malla"]))
            return libres[0]

        agente = Agente(self.medidor(malla, nops, lambda c: 0.1 * c["filas"] * c["columnas"]), BASE,
                        parametros(malla=[4, 5], epsilon=1.0, paciencia=50, max_episodios=5, historial_llm=2),
                        eleccion_rl=rl)
        agente.correr()
        self.assertTrue(vistos)
        for episodio, historial, mejor, c_malla in vistos:
            self.assertEqual(historial, list(range(max(0, episodio - 2), episodio)))
            self.assertEqual(c_malla, 5.0)
            if episodio == 0:
                self.assertIsNone(mejor)
        self.assertEqual({e for e, *_ in vistos}, set(range(5)))

    def test_5_infraestructura_no_cuenta_para_la_paciencia(self):
        """Un fallo de infraestructura no tiene recompensa, no se guarda en la caché y no consume paciencia."""
        malla, nops = self.malla_y_nops()
        agente = Agente(self.medidor(malla, nops, lambda c: None, estado_rect="error_infraestructura"), BASE,
                        parametros(malla=[4, 5], paciencia=2, max_episodios=4))
        historial = agente.correr()
        self.assertEqual(len(historial), 4)
        self.assertEqual(agente.motivo_fin, "max_episodios")
        self.assertEqual({ep["recompensa"] for ep in historial}, {None})
        self.assertEqual(len(self.llamadas), 1 + 4)  # la malla y un reintento de simulación por episodio
        self.assertEqual(agente.cache_episodios, {})
        self.assertIsNone(agente.mejor)
        self.assertEqual(agente.episodio, historial[-1])

    def test_7_curva_y_gif_de_episodios(self):
        """gif_camino.py dibuja curva_episodios.png y episodios.gif (un cuadro por episodio) desde
        episodios.json, igual byte por byte en dos corridas."""
        import hashlib
        import gif_camino
        malla, nops = self.malla_y_nops()
        costo = lambda c: 0.1 * c["filas"] * c["columnas"]
        agente = Agente(self.medidor(malla, nops, costo), BASE,
                        parametros(malla=[4, 5], epsilon=1.0, semilla=7, paciencia=4, max_episodios=6))
        agente.correr()
        datos = {"epsilon": 1.0, "C_malla": agente.evaluacion.costo, "C_best": agente.c_best,
                 "motivo_fin": agente.motivo_fin, "episodios": agente.historial}
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp)
            (corrida / "historia.json").write_text(json.dumps(agente.historia()), encoding="utf-8")
            (corrida / "episodios.json").write_text(json.dumps(datos), encoding="utf-8")
            hashes = []
            for salida in ("a", "b"):
                with mock.patch("sys.stdout"):
                    self.assertEqual(gif_camino.main([str(corrida), "--salida", str(corrida / salida)]), 0)
                hashes.append([hashlib.sha256((corrida / salida / n).read_bytes()).hexdigest()
                               for n in ("curva_episodios.png", "episodios.gif")])
            self.assertEqual(hashes[0], hashes[1])
            from PIL import Image
            with Image.open(corrida / "a" / "episodios.gif") as g:
                self.assertEqual(g.n_frames, len(agente.historial))
            with mock.patch("sys.stdout"):
                gif_camino.main([str(corrida), "--salida", str(corrida / "c"), "--sin-episodios"])
            self.assertFalse((corrida / "c" / "curva_episodios.png").exists())

    def test_6_prompt_con_el_historial(self):
        """El prompt del LLM incluye los episodios anteriores con su C real y el mejor."""
        ep = {"n": 0, "camino": ["PE00", "PE01", "PE11"], "malla": ["M E M", "M V M"], "filas": 2, "columnas": 3,
              "estado": "correcta", "repetido": False, "C": 1.84, "C_estimado": 0.69, "recompensa": 3.16}
        contexto = {"episodio": 1, "historial": [ep], "mejor": ep, "C_malla": 5.0}
        args = TestEleccionLLM.ARGS[:6]
        prompt = construir_prompt(*args, contexto)
        self.assertIn("Episodio 1", prompt)
        self.assertIn("ep 0: camino PE00→PE01→PE11; rectángulo M E M / M V M (2x3), correcta, C real 1.84 "
                      "(estimado 0.69), R = 3.16", prompt)
        self.assertIn("Mejor hasta ahora: ep 0", prompt)
        self.assertIn("C de la malla completa: 5", prompt)
        self.assertNotIn("Episodios anteriores", construir_prompt(*args))  # el primer episodio no trae historial
        self.assertNotIn("Episodios anteriores", construir_prompt(*args, {"episodio": 0, "historial": [],
                                                                           "mejor": None, "C_malla": 5.0}))


class TestSeguirOParar(unittest.TestCase):
    """En el punto de paro, con probabilidad epsilon, el agente RL decide PARAR o SEGUIR n PEs más:
    así puede probar configuraciones más grandes que las que prefiere el estimado."""

    def agente(self, eleccion, **cambios):
        malla = Agente(None, BASE, parametros(malla=[4, 5])).malla()
        nops = {"op": nops_iguales(malla, 0)}
        datos = dict(malla=[4, 5], epsilon=1.0, max_extension=5)
        datos.update(cambios)
        return Agente(lambda c: evaluacion_falsa(malla, nops), BASE, parametros(**datos), eleccion_rl=eleccion)

    def rl_con(self, decisiones):
        """Un agente RL que, en el punto de paro, responde las ``decisiones`` en orden (y luego 0), y
        en los pasos normales elige como el greedy (el vecino de menor C)."""
        pendientes, llamadas = list(decisiones), []

        def rl(actual, libres, pesos, camino, opciones, c_actual, contexto, parada=None):
            if parada:
                llamadas.append((list(camino), contexto.get("rectangulo"), parada))
                return pendientes.pop(0) if pendientes else 0
            return min(libres, key=lambda v: (opciones[v]["C"], libres.index(v)))
        rl.llamadas = llamadas
        return rl

    def test_1_seguir_n_agrega_n_pes(self):
        """PARAR deja el camino del greedy; SEGUIR 3 agrega exactamente 3 PEs más y luego se revisa el
        paro otra vez (ahí el RL dice PARAR)."""
        base = self.agente(self.rl_con([0]))
        camino_base = base.buscar()
        self.assertEqual([(x["decision"], x["n"]) for x in base.extensiones], [("rl", 0)])
        rl = self.rl_con([3, 0])
        agente = self.agente(rl)
        camino = agente.buscar()
        self.assertEqual(camino[:len(camino_base)], camino_base)
        self.assertEqual(len(camino), len(camino_base) + 3)
        self.assertEqual([(x["decision"], x["n"], x["pes"]) for x in agente.extensiones],
                         [("rl", 3, len(camino_base)), ("rl", 0, len(camino_base) + 3)])
        self.assertEqual(agente.parada, "sin_mejora")
        # el RL ve el rectángulo que se simularía si para ahora y el máximo permitido
        camino_visto, rectangulo, maximo = rl.llamadas[0]
        self.assertEqual(camino_visto, camino_base)
        self.assertEqual(rectangulo, [" ".join(f) for f in agente.malla_episodio(camino_base)])
        self.assertEqual(maximo, 5)

    def test_2_fuera_de_rango_es_parar(self):
        """Una respuesta fuera de 0..max_extension (o que no es un entero) cuenta como rl_invalida: para."""
        for respuesta in (6, -1, "PE01", True, None):
            agente = self.agente(lambda *a, parada=None, r=respuesta: r if parada else a[1][0])
            agente.buscar()
            self.assertEqual([(x["decision"], x["n"]) for x in agente.extensiones], [("rl_invalida", 0)], respuesta)

    def test_3_max_extension_0_y_epsilon_0_siempre_paran(self):
        rl = self.rl_con([5])
        agente = self.agente(rl, max_extension=0)
        agente.buscar()
        self.assertEqual((agente.extensiones, rl.llamadas), ([], []))
        rl = self.rl_con([5])
        agente = self.agente(rl, epsilon=0.0)
        agente.buscar()
        self.assertEqual((agente.extensiones, rl.llamadas), ([], []))

    def test_4_sin_rl_el_azar_decide(self):
        """Sin agente RL, el azar elige de 0 a max_extension (exploracion), con la semilla."""
        agente = self.agente(None, semilla=4)
        camino = agente.buscar()
        self.assertTrue(agente.extensiones)
        self.assertEqual({x["decision"] for x in agente.extensiones}, {"exploracion"})
        self.assertTrue(all(0 <= x["n"] <= 5 for x in agente.extensiones))
        self.assertEqual(camino, self.agente(None, semilla=4).buscar())

    def test_5_el_episodio_guarda_la_extension(self):
        """El episodio (y el historial que ve el RL) dice cuántos PEs siguió tras el punto de paro."""
        agente = self.agente(self.rl_con([2, 0]), max_episodios=1)
        agente.correr()
        ep = agente.historial[0]
        self.assertEqual(ep["extension"], 2)
        self.assertEqual([x["n"] for x in ep["extensiones"]], [2, 0])
        self.assertEqual(agente.resultado()["extensiones"], ep["extensiones"])
        self.assertEqual(agente.historia()["extensiones"], ep["extensiones"])

    def test_6_prompt_y_lectura_de_parar_o_seguir(self):
        """El prompt del punto de paro pide PARAR o SEGUIR n; solo se aceptan esas respuestas exactas."""
        contexto = {"episodio": 1, "historial": [], "mejor": None, "C_malla": 29.6,
                    "rectangulo": ["M E M", "M V M"]}
        prompt = construir_prompt_parada("PE11", ["PE00", "PE01", "PE11"], 0.687, contexto, 10)
        self.assertIn("PARAR", prompt)
        self.assertIn("SEGUIR n", prompt)
        self.assertIn("entre 1 y 10", prompt)
        self.assertIn("rectángulo M E M / M V M (2x3)", prompt)
        self.assertIn("C estimado es 0.687", prompt)
        for respuesta, esperado in (("PARAR", 0), ("parar\n", 0), ("`PARAR`", 0), ("PARAR.", 0),
                                    ("SEGUIR 3", 3), ("seguir 10", 10), (" `SEGUIR 1`. ", 1)):
            self.assertEqual(interpretar_parada(respuesta, 10), esperado, respuesta)
        for respuesta in ("SEGUIR 11", "SEGUIR 0", "SEGUIR", "SEGUIR -2", "Creo que SEGUIR 3", "PARAR o SEGUIR 2",
                          "PE01", "", None, "SEGUIR 3 PEs"):
            self.assertIsNone(interpretar_parada(respuesta, 10), respuesta)

    def test_7_el_llm_por_cli_en_el_punto_de_paro(self):
        """EleccionLLM con ``parada`` usa el prompt del punto de paro y lee PARAR / SEGUIR n."""
        eleccion = EleccionLLM(["claude", "-p", "{prompt}"])
        args = TestEleccionLLM.ARGS[:6] + ({"episodio": 0, "historial": [], "mejor": None, "C_malla": 29.6,
                                             "rectangulo": ["M E M", "M V M"]},)
        with mock.patch("dijkstra.rl_llm.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout="SEGUIR 4\n", stderr="")) as run:
            self.assertEqual(eleccion(*args, parada=10), 4)
        self.assertIn("PARAR", run.call_args.args[0][2])
        with mock.patch("dijkstra.rl_llm.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout="PE01", stderr="")):
            self.assertIsNone(eleccion(*args, parada=10))


class TestPareto(unittest.TestCase):
    """herramientas/pareto.py: rejilla de rectángulos, frente de Pareto área-latencia y gráfica, con un
    evaluador falso (no simula)."""

    @staticmethod
    def medidor():
        """Área = PEs ponderados; latencia que baja con más PEs; una configuración dominada a propósito."""
        llamadas = []

        def medir(conf):
            llamadas.append(conf)
            malla = tuple(tuple(f.split()) for f in conf["malla"])
            e = evaluacion_falsa(malla, {"op": nops_iguales(malla, 0)})
            area = sum(W[t] for f in malla for t in f)
            pes = len(malla) * len(malla[0])
            latencia = 4.0 / pes + (0.5 if conf["malla"] == ["M E E M", "M V V M", "M E E M"] else 0.0)
            e.metricas = {"C_A": area, "C_A_norm": area / 76, "C_L_norm": latencia, "C_R": 0.0}
            e.costo = area / 76 + latencia
            return e
        medir.llamadas = llamadas
        return medir

    def test_1_rango_y_rectangulos(self):
        import pareto
        self.assertEqual(pareto.rango("2-5"), [2, 3, 4, 5])
        self.assertEqual(pareto.rango("3"), [3])
        with self.assertRaises(ValueError):
            pareto.rango("5-2")
        malla = Agente(None, BASE, parametros(malla=[6, 8])).malla()
        self.assertEqual(pareto.rectangulo(malla, "M", 2, 1), (("M", "E", "M"), ("M", "V", "M")))
        self.assertEqual(pareto.nombre(pareto.rectangulo(malla, "M", 3, 4)), "3x6")
        with self.assertRaises(ValueError):
            pareto.rectangulo(malla, "M", 2, 20)

    def test_2_frente(self):
        """Eficiente: ninguna otra tiene menor o igual área y latencia (y menor en alguna)."""
        import pareto
        p = [{"n": "a", "C_A": 1, "C_L_norm": 5}, {"n": "b", "C_A": 2, "C_L_norm": 3},
             {"n": "c", "C_A": 3, "C_L_norm": 4}, {"n": "d", "C_A": 4, "C_L_norm": 1},
             {"n": "e", "C_A": 2, "C_L_norm": 3}]
        self.assertEqual([x["n"] for x in pareto.frente(p)], ["a", "b", "e", "d"])  # c está dominada por b

    def test_3_evalua_reusa_y_grafica(self):
        """Evalúa la rejilla, guarda pareto.json/.csv/.png, marca eficientes y dominadas; la segunda vez
        no simula nada (retoma de pareto.json), y --solo-grafica redibuja igual byte por byte."""
        import hashlib
        import pareto
        medir = self.medidor()
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp)
            with mock.patch("sys.stdout"):
                self.assertEqual(pareto.main([str(corrida), "--filas", "2-3", "--interiores", "1-2"], medir=medir), 0)
            self.assertEqual(len(medir.llamadas), 4)
            datos = json.loads((corrida / "pareto_rejilla.json").read_text())
            conf = {c["nombre"]: c for c in datos["configuraciones"]}
            self.assertEqual(set(conf), {"2x3", "2x4", "3x3", "3x4"})
            self.assertFalse(conf["3x4"]["eficiente"])  # dominada a propósito
            self.assertTrue(conf["2x3"]["eficiente"])
            self.assertTrue((corrida / "pareto_rejilla.csv").read_text().startswith("configuracion,"))
            png = hashlib.sha256((corrida / "frente_pareto_rejilla.png").read_bytes()).hexdigest()
            with mock.patch("sys.stdout"):
                pareto.main([str(corrida), "--filas", "2-3", "--interiores", "1-2"],
                            medir=lambda c: (_ for _ in ()).throw(AssertionError("no debía simular")))
                pareto.main([str(corrida), "--solo-grafica"])
            self.assertEqual(hashlib.sha256((corrida / "frente_pareto_rejilla.png").read_bytes()).hexdigest(), png)

    def test_4_listar_no_simula(self):
        import pareto
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stdout"):
                self.assertEqual(pareto.main([tmp, "--listar"], medir=lambda c: 1 / 0), 0)
            self.assertFalse((Path(tmp) / "pareto_rejilla.json").exists())


class TestModoBloques(unittest.TestCase):
    """Modo "bloques": descenso desde la malla de referencia quitando bloques de filas o columnas; cada
    malla propuesta se simula y se acepta solo si su C real baja (evaluador falso)."""

    MALLA = [8, 10]

    def referencia(self):
        return Agente(None, BASE, parametros(malla=self.MALLA)).malla()

    def medidor(self):
        """C = área/76 + 30/PEs: quitar PEs baja el área pero sube la latencia."""
        referencia = self.referencia()
        llamadas = []

        def medir(conf):
            llamadas.append(f"{conf['filas']}x{conf['columnas']}")
            m = tuple(tuple(f.split()) for f in conf["malla"])
            base = referencia if len(llamadas) == 1 else m
            e = evaluacion_falsa(base, {"op": nops_iguales(base, 1)})
            area, pes = sum(W[x] for f in m for x in f), len(m) * len(m[0])
            e.metricas = {"C_A": area, "C_A_norm": area / 76, "C_L_norm": 30 / pes, "C_R": 0.0}
            e.costo = area / 76 + 30 / pes
            return e
        medir.llamadas = llamadas
        return medir

    def agente(self, eleccion=None, **cambios):
        datos = dict(malla=self.MALLA, modo="bloques", bloques=[4, 2, 1], max_episodios=3, paciencia=2)
        datos.update(cambios)
        return Agente(self.medidor(), BASE, parametros(**datos), eleccion_rl=eleccion)

    def test_1_rectangulos(self):
        from dijkstra.bloques import Descenso, Estado
        a = Agente(None, BASE, parametros(malla=self.MALLA))
        d = Descenso(a)
        e0 = d.inicial()
        self.assertEqual(d.malla(e0), self.referencia())
        self.assertEqual((e0.filas, e0.interiores), (8, 8))
        e = d.aplicar(e0, "FILAS INICIO", 4)
        self.assertEqual(d.malla(e)[0], ("M",) + ("E",) * 8 + ("M",))
        self.assertEqual(d.descripcion(d.aplicar(e0, "COLUMNAS FIN", 4)), "8x6")
        self.assertIsNone(d.aplicar(e0, "FILAS FIN", 8))  # no quedan filas
        self.assertFalse(d.valido(Estado(0, 1, 0, 8)))  # una sola fila E: sin PE vectorial
        self.assertTrue(d.valido(Estado(0, 2, 0, 1)))  # 2x3: M E M / M V M
        for fila in d.malla(Estado(2, 5, 3, 6)):
            self.assertEqual((fila[0], fila[-1]), ("M", "M"))

    def test_2_descenso_acepta_si_baja_el_costo_real(self):
        """Parte de la referencia sin volver a simularla; cada malla aceptada baja el C real; las que no,
        se descartan; termina cuando ni el bloque más pequeño mejora."""
        a = self.agente(max_episodios=1)
        a.correr()
        ep = a.historial[0]
        self.assertEqual(a.medir.llamadas.count("8x10"), 1)  # la referencia, una sola vez
        self.assertEqual(ep["modo"], "bloques")
        self.assertEqual(ep["trayectoria"][0]["accion"], "inicio")
        aceptadas = [t for t in ep["trayectoria"] if t.get("aceptada")]
        costos = [t["C"] for t in aceptadas]
        self.assertEqual(costos, sorted(costos, reverse=True))
        self.assertEqual(len(set(costos)), len(costos))
        self.assertEqual(ep["parada"], "sin_mejora")
        self.assertGreater(ep["simuladas"], ep["aceptadas"] - 1)  # hubo descartadas
        self.assertEqual(ep["C"], costos[-1])
        self.assertEqual((ep["filas"], ep["columnas"]), (aceptadas[-1]["filas"], aceptadas[-1]["columnas"]))
        claves = {a._clave(a.configuracion(tuple(tuple(f.split()) for f in ep["malla"])))}
        self.assertTrue(claves <= set(a.cache_episodios))
        # las acciones que darían la misma malla no se prueban dos veces
        self.assertEqual(len(a.medir.llamadas), len(set(a.medir.llamadas)))
        self.assertEqual(a.resultado()["modo"], "bloques")
        json.dumps(a.historial, allow_nan=False)

    def test_3_episodios_reusan_la_cache(self):
        """Con epsilon 0 todos los descensos son iguales: el segundo no simula nada y la paciencia termina."""
        a = self.agente(max_episodios=5, paciencia=2)
        a.correr()
        self.assertEqual(a.motivo_fin, "paciencia")
        self.assertEqual(len(a.historial), 3)
        self.assertGreater(a.historial[0]["nuevas"], 0)
        self.assertEqual([ep["nuevas"] for ep in a.historial[1:]], [0, 0])
        self.assertTrue(a.historial[0]["mejora"])
        self.assertTrue(a.historial[1]["repetido"])

    def test_4_el_agente_rl_decide_la_accion(self):
        """El LLM puede elegir otra acción, BAJAR de bloque o PARAR; una respuesta inválida usa Greedy."""
        class RL:
            def __init__(self, respuestas):
                self.respuestas, self.contextos = list(respuestas), []

            def decidir_bloque(self, contexto):
                self.contextos.append(contexto)
                return self.respuestas.pop(0) if self.respuestas else None

        rl = RL(["COLUMNAS FIN"])
        a = self.agente(rl, epsilon=1.0, max_episodios=1)
        a.correr()
        t = a.historial[0]["trayectoria"][1]
        self.assertEqual((t["accion"], t["quien"]), ("COLUMNAS FIN", "rl"))
        ctx = rl.contextos[0]
        self.assertEqual((ctx["modo"], ctx["bloque"], ctx["malla_actual"]), ("bloques", 4, "8x10"))
        # quitar 4 filas (par) del inicio o del final, o columnas de cualquier lado, da la misma malla: se
        # ofrece una sola, la de FIN (el rectángulo queda anclado en PE00)
        self.assertIn("FILAS FIN", ctx["opciones"])
        self.assertNotIn("FILAS INICIO", ctx["opciones"])
        self.assertNotIn("COLUMNAS INICIO", ctx["opciones"])
        self.assertEqual(ctx["trayectoria"][0]["accion"], "inicio")
        self.assertTrue(all(d == "rl_invalida" for d in a.historial[0]["decisiones"][1:]))  # luego None

        a = self.agente(RL(["PARAR"]), epsilon=1.0, max_episodios=1)
        a.correr()
        self.assertEqual(a.historial[0]["parada"], "rl_parar")
        self.assertEqual((a.historial[0]["filas"], a.historial[0]["columnas"]), (8, 10))

        a = self.agente(RL(["BAJAR", "BAJAR", "PARAR"]), epsilon=1.0, max_episodios=1)
        a.correr()
        self.assertEqual([t["accion"] for t in a.historial[0]["trayectoria"][1:]], ["BAJAR", "BAJAR", "PARAR"])

    def test_5_prompt_y_lectura(self):
        from dijkstra.rl_llm import construir_prompt_bloques, interpretar_bloque
        contexto = {"episodio": 0, "historial": [], "mejor": None, "C_malla": 29.6, "malla_actual": "16x32",
                    "C_actual": 14.8, "bloque": 16, "bloques": [16, 8, 4, 2, 1],
                    "trayectoria": [{"accion": "inicio", "filas": 32, "columnas": 32, "C": 29.6}],
                    "opciones": {"COLUMNAS INICIO": {"tamano": "16x16", "C_estimado": 7.3}}}
        prompt = construir_prompt_bloques(contexto)
        for texto in ("16x32", "C real 14.8", "Bloque actual: 16", "COLUMNAS INICIO: queda 16x16", "BAJAR", "PARAR"):
            self.assertIn(texto, prompt)
        opciones = {"FILAS FIN": None, "COLUMNAS INICIO": None}
        for respuesta, esperado in (("FILAS FIN", "FILAS FIN"), ("filas  fin\n", "FILAS FIN"),
                                    ("`COLUMNAS INICIO`.", "COLUMNAS INICIO"), ("BAJAR", "BAJAR"), ("parar", "PARAR")):
            self.assertEqual(interpretar_bloque(respuesta, opciones), esperado)
        for respuesta in ("FILAS INICIO", "quito FILAS FIN", "FILAS", "", None, "SEGUIR 3"):
            self.assertIsNone(interpretar_bloque(respuesta, opciones), respuesta)

    def test_6_curvas_gif_y_pareto_de_la_exploracion(self):
        """gif_camino dibuja curva_exploracion.png y exploracion.gif (un cuadro por malla probada) y
        pareto --exploracion arma el frente con las configuraciones simuladas, sin simular."""
        import hashlib
        import gif_camino
        import pareto
        a = self.agente(max_episodios=2, paciencia=5)
        a.correr()
        datos = {"modo": "bloques", "malla_referencia": [" ".join(f) for f in self.referencia()],
                 "epsilon": 0.0, "C_malla": a.evaluacion.costo, "C_best": a.c_best,
                 "motivo_fin": a.motivo_fin, "episodios": a.historial}
        with tempfile.TemporaryDirectory() as tmp:
            corrida = Path(tmp)
            (corrida / "episodios.json").write_text(json.dumps(datos), encoding="utf-8")
            (corrida / "evaluaciones_episodios.json").write_text(json.dumps({"episodios": [
                {"configuracion": json.loads(k), "evaluacion": e.como_dict()} for k, e in a.cache_episodios.items()]}))
            hashes = []
            for salida in ("a", "b"):
                with mock.patch("sys.stdout"):
                    self.assertEqual(gif_camino.main([str(corrida), "--salida", str(corrida / salida)]), 0)
                hashes.append([hashlib.sha256((corrida / salida / n).read_bytes()).hexdigest()
                               for n in ("curva_costos.png", "exploracion.gif", "curva_episodios.png")])
            self.assertEqual(hashes[0], hashes[1])
            from PIL import Image
            cuadros = gif_camino.cuadros_exploracion(datos)
            with Image.open(corrida / "a" / "exploracion.gif") as g:
                self.assertEqual(g.n_frames, len(cuadros) + 1)  # más el cuadro final
            # por cada malla probada: un cuadro de propuesta y uno de resultado
            probadas = gif_camino.probadas(gif_camino.pasos_exploracion(datos)) - 1
            self.assertEqual(sum(c["tipo"] == "propuesta" for c in cuadros), probadas)
            self.assertEqual(sum(c["tipo"] in ("aceptada", "descartada") for c in cuadros), probadas)
            with mock.patch("sys.stdout"):
                self.assertEqual(pareto.main([str(corrida), "--exploracion"]), 0)
            frente = json.loads((corrida / "configuraciones.json").read_text())["configuraciones"]
            self.assertEqual(len(frente), len(a.cache_episodios))
            self.assertTrue(any(c["eficiente"] for c in frente))
            self.assertTrue(all(c["orden"] is not None for c in frente))
            self.assertTrue((corrida / "frente_pareto.png").exists())
            filas = (corrida / "configuraciones.csv").read_text().splitlines()
            self.assertEqual(len(filas), len(frente) + 1)
            self.assertIn("op_ciclos", filas[0])

    def test_7_fase_tipos(self):
        """Con tipos: true, después del descenso se cambian PEs del interior E <-> V (los bordes M no se
        tocan); cada malla se simula y se acepta si su C real baja. Con C = área/76 + 30/PEs, pasar de V
        a E baja el área: la fase deja un solo V (hace falta al menos un escalar y un vectorial)."""
        sin = self.agente(max_episodios=1)
        sin.correr()
        a = self.agente(max_episodios=1, tipos=True)
        a.correr()
        ep = a.historial[0]
        tipos = [t for t in ep["trayectoria"] if t.get("fase") == "tipos"]
        self.assertTrue(tipos)
        self.assertEqual(ep["parada_tipos"], "sin_mejora")
        self.assertGreater(ep["cambios_tipo"], 0)
        self.assertLess(ep["C"], sin.historial[0]["C"])  # la fase de tipos mejora la malla del descenso
        final = [f.split() for f in ep["malla"]]
        self.assertEqual((len(final), len(final[0])), (sin.historial[0]["filas"], sin.historial[0]["columnas"]))
        self.assertTrue(all(f[0] == "M" and f[-1] == "M" for f in final))
        self.assertEqual(sum(f[1:-1].count("V") for f in final), 1)
        for t in tipos:
            self.assertNotIn(t["pe"][1], (0, len(final[0]) - 1))  # nunca una columna de memoria
            self.assertEqual({t["de"], t["a"]}, {"E", "V"})
        aceptadas = [t["C"] for t in tipos if t["aceptada"]]
        self.assertEqual(aceptadas, sorted(aceptadas, reverse=True))
        # ninguna malla se simula dos veces en la fase
        self.assertEqual(len({tuple(t["malla"]) for t in tipos}), len(tipos))
        json.dumps(a.historial, allow_nan=False)

        # el agente RL puede elegir el cambio o PARAR la fase
        class RL:
            def __init__(self, respuestas):
                self.respuestas, self.contextos = list(respuestas), []

            def decidir_bloque(self, contexto):
                return "PARAR"

            def decidir_tipo(self, contexto):
                self.contextos.append(contexto)
                return self.respuestas.pop(0) if self.respuestas else "PARAR"

        otra = "CAMBIAR 0 1"  # el descenso para de inmediato (PARAR): la fase empieza en la referencia
        rl = RL([otra])
        b = self.agente(rl, epsilon=1.0, max_episodios=1, tipos=True)
        b.correr()
        ep = b.historial[0]
        ctx = rl.contextos[0]
        self.assertEqual(ctx["modo"], "tipos")
        self.assertIn(otra, ctx["opciones"])
        t = [t for t in ep["trayectoria"] if t.get("fase") == "tipos"]
        self.assertEqual((t[0]["accion"], t[0]["quien"]), (otra, "rl"))
        self.assertEqual(t[-1]["accion"], "PARAR")
        self.assertEqual(ep["parada_tipos"], "rl_parar")

        # el GIF dibuja los tipos cambiados de la malla
        import gif_camino
        # rectángulos anclados en PE00: solo cuentan el tamaño y la paridad de la fila inicial
        self.assertEqual(gif_camino.anclar([16, 32, 8, 30]), [0, 16, 0, 22])
        self.assertEqual(gif_camino.anclar([5, 8, 3, 5]), [1, 4, 0, 2])
        self.assertEqual(gif_camino.accion_vista([0, 32, 0, 30], [16, 32, 0, 30]), "FILAS FIN")
        self.assertEqual(gif_camino.accion_vista([0, 4, 0, 2], [1, 4, 0, 2]), "FILAS INICIO")
        self.assertEqual(gif_camino.accion_vista([0, 4, 0, 2], [0, 4, 1, 2]), "COLUMNAS FIN")
        datos = {"modo": "bloques", "malla_referencia": [" ".join(f) for f in self.referencia()],
                 "epsilon": 0.0, "C_malla": a.evaluacion.costo, "C_best": a.c_best,
                 "motivo_fin": a.motivo_fin, "episodios": a.historial}
        cuadros = gif_camino.cuadros_exploracion(datos)
        self.assertTrue(any(c.get("fase") == "tipos" for c in cuadros))
        malla = gif_camino.MallaPEs(datos["malla_referencia"])
        ultimo = [c for c in cuadros if c["tipo"] == "aceptada"][-1]
        activos = malla.activos(ultimo["rect"], ultimo["malla"])
        self.assertEqual(sum(x == "V" for x in activos.values()), 1)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gif_camino.gif_exploracion(datos, Path(tmp) / "e.gif"), len(cuadros) + 1)

        # el prompt y la lectura de la respuesta
        from dijkstra.rl_llm import construir_prompt_tipos, interpretar_tipo
        prompt = construir_prompt_tipos(ctx)
        for texto in ("escalar (E) o vectorial (V)", "fila 0:", otra, "PARAR"):
            self.assertIn(texto, prompt)
        self.assertEqual(interpretar_tipo(f" `{otra.lower()}`.", ctx["opciones"]), otra)
        self.assertIsNone(interpretar_tipo("CAMBIAR 99 99", ctx["opciones"]))
        self.assertIsNone(interpretar_tipo("FILAS FIN", ctx["opciones"]))


class TestResultados(unittest.TestCase):
    """Lectura completa del reporte de ciclos y tabla de resultados (dijkstra/tabla.py)."""

    REPORTE = """Total de ciclos:        154
Ciclos de cómputo:      77 (50.0%)
Ciclos de comunicación: 76 (49.4%)
Ciclos inactivos:       1 (0.6%)

Instrucciones por PE (memoria = LD/ST/VLD/VST, cálculo = ADD..DIV, V.. y VREDSUM,
comunicación = SEND/RECV/VSEND/VRECV, otras = MOV/VBCAST):
  PE    Tipo     Memoria  Cálculo  Comunic.  Otras    NOP  Ocupado
  PE00  memoria       26        0        26      0    102    33.8%
  PE01  computo        0       75        76      1      2    98.7%
  PE02  memoria       50        0        50      0     54    64.9%

Instrucciones que ejecuta cada PE (sin NOP):
  PE00 (memoria): ST x1, RECV x1, VLD x25, VSEND x25
  PE01 (computo): MOV x1, ADD x25, SEND x1, VADD x25, VREDSUM x25, VSEND x25, VRECV x50
  PE02 (memoria): VLD x25, VST x25, VSEND x25, VRECV x25

Accesos a memoria (posiciones dentro de cada región, y cuántas palabras lee o escribe en total):
  PE00 (memoria): lee a[0..99] (100 lecturas); escribe result[0] (1 escritura)
  PE02 (memoria): lee b[0..99] (100 lecturas); escribe c[0..99] (100 escrituras)

Pasos de comunicación entre PEs:
  Ciclo   1: PE00 -> PE01 (este)  VSEND east, v0
  Ciclo   2: PE02 -> PE01 (oeste)  VSEND west, v1
"""

    def test_1_detalle_del_reporte(self):
        from dijkstra.simulacion import leer_detalle, leer_reporte
        d = leer_detalle(self.REPORTE)
        self.assertEqual((d["ciclos_computo"], d["ciclos_comunicacion"], d["ciclos_inactivos"]), (77, 76, 1))
        self.assertEqual((d["instrucciones_memoria"], d["instrucciones_calculo"], d["instrucciones_comunicacion"],
                          d["instrucciones_otras"], d["nops"]), (76, 75, 152, 1, 158))
        self.assertEqual(d["instrucciones"] + d["nops"], 154 * 3)  # ranuras = ciclos · PEs
        self.assertEqual(sum(d["por_instruccion"].values()), d["instrucciones"])
        self.assertEqual(d["por_instruccion"]["VRECV"], 75)
        self.assertEqual((d["lecturas"], d["escrituras"], d["transferencias"]), (200, 101, 2))
        self.assertEqual(leer_reporte(self.REPORTE)[:2], (154, 158))
        vacio = leer_detalle("Total de ciclos: 3")
        self.assertIsNone(vacio["ciclos_computo"])
        self.assertIsNone(vacio["por_instruccion"])

    def test_2b_nombres_con_la_distribucion_de_tipos(self):
        """El nombre muestra dónde quedó cada tipo de PE (interior, fila por fila); las grandes, el tamaño."""
        from dijkstra.tabla import nombre_malla, patron
        self.assertEqual(nombre_malla(["M V E M", "M V V M"]), "2x4 [VE/VV]")
        self.assertEqual(nombre_malla((("M", "E", "M"), ("M", "V", "M"))), "2x3 [E/V]")
        self.assertEqual(patron(["M V V M", "M V E M", "M V V M"]), "VV/VE/VV")
        grande = ["M " + " ".join("E" * 14) + " M"] * 16
        self.assertIsNone(patron(grande))
        self.assertEqual(nombre_malla(grande), "16x16")

    def test_2_tabla(self):
        """Cada configuración con sus costos y, por operación, ciclos, instrucciones, NOPs, latencia y error;
        el frente de Pareto marcado; sin detalle del reporte, las instrucciones salen de ranuras − NOPs."""
        from dijkstra import tabla
        from dijkstra.simulacion import leer_detalle
        con = EvaluacionOperacion("op", "correcta", ciclos=154, nops=158, slots=462, C_L=100.0, C_L_norm=0.5,
                                  C_R=0.0, costo=1.2, detalle=leer_detalle(self.REPORTE),
                                  validacion={"total_elementos": 101, "elementos_fallidos": 0})
        sin = EvaluacionOperacion("op", "correcta", ciclos=74, nops=74029, slots=75776, C_L_norm=1.0, C_R=0.0)
        chica = (("M", "E", "M"), ("M", "V", "M"))
        grande = tuple(("M",) + ("E",) * 30 + ("M",) for _ in range(32))
        e1 = Evaluacion(["M E M", "M V M"], "correcta", 1.2, {"C_A": 51, "C_A_norm": 0.67, "C_L_norm": 1.76, "C_R": 0.0},
                        operaciones={"op": con})
        e2 = Evaluacion([], "correcta", 29.6, {"C_A": 2208, "C_A_norm": 29.05, "C_L_norm": 0.55, "C_R": 0.0},
                        operaciones={"op": sin})
        registros = tabla.construir([(grande, e2, "referencia"), (chica, e1, "exploracion"), (chica, e1, "exploracion")])
        # sin repetir, por área; las mallas pequeñas llevan la distribución de tipos del interior
        self.assertEqual([r["nombre"] for r in registros], ["2x3 [E/V]", "32x32"])
        r = registros[0]
        self.assertEqual((r["n_M"], r["n_E"], r["n_V"], r["pes"]), (4, 1, 1, 6))
        o = r["operaciones"]["op"]
        self.assertEqual((o["ciclos"], o["instrucciones"], o["nops"], o["ciclos_comunicacion"]), (154, 304, 158, 76))
        self.assertEqual((o["elementos"], o["elementos_fallidos"]), (101, 0))
        self.assertEqual(registros[1]["operaciones"]["op"]["instrucciones"], 75776 - 74029)
        self.assertTrue(all(x["eficiente"] for x in registros))  # menos área contra menos latencia
        with tempfile.TemporaryDirectory() as tmp:
            tabla.escribir(tmp, registros)
            ancho = (Path(tmp) / "configuraciones.csv").read_text().splitlines()
            largo = (Path(tmp) / "configuraciones_operaciones.csv").read_text().splitlines()
            self.assertEqual(len(ancho), 3)
            self.assertIn("op_ciclos_comunicacion", ancho[0])
            self.assertIn("inst_VRECV", largo[0])
            self.assertEqual(len(json.loads((Path(tmp) / "configuraciones.json").read_text())["configuraciones"]), 2)


class TestReporteCorrida(unittest.TestCase):
    """reporte_corrida: tiempos, episodios, decisiones (Greedy y LLM), Dijkstra, configuraciones de menor y
    mayor costo, y recursos computacionales."""

    def test_1_reporte(self):
        from datetime import timedelta
        from dijkstra import reporte_corrida, tabla
        modo = TestModoBloques()
        a = modo.agente(max_episodios=2, paciencia=5, epsilon=0.5, semilla=1)
        a.correr()
        evaluaciones = [(tuple(tuple(f.split()) for f in json.loads(k)["malla"]), e, "exploracion")
                        for k, e in a.cache_episodios.items()]
        for _, e, _ in evaluaciones:  # tiempos de Dijkstra como los que registra el evaluador
            for o in e.operaciones.values():
                o.tiempo_mapeo_s, o.tiempo_validacion_s, o.tiempo_simulacion_s = 1.5, 0.1, 0.4
                o.mapeo = {"operaciones": 10, "transferencias": 3}
        registros = tabla.construir(evaluaciones, a.historial)
        inicio = reporte_corrida.ahora()
        llm = [{"tipo": "bloque", "segundos": 4.0, "timeout": False, "valida": True, "respuesta": "FILAS FIN"},
               {"tipo": "bloque", "segundos": None, "timeout": True, "valida": False, "respuesta": None}]
        monitor = reporte_corrida.MonitorRecursos(intervalo=0.05).iniciar()
        rep = reporte_corrida.construir(vars(a.parametros), {"exploración": 12.0}, inicio, inicio + timedelta(seconds=90),
                                        a.historial, a.motivo_fin, a.c_best,
                                        [{"configuracion": "4x10", "pes": 40, "estado": "correcta", "C": 2.6, "segundos": 7.0}],
                                        llm, registros, evaluaciones, reporte_corrida.uso_cpu(90.0), monitor.detener())
        self.assertEqual(rep["general"]["duracion_s"], 90.0)
        self.assertEqual(rep["episodios"]["total"], len(a.historial))
        d = rep["decisiones"]
        self.assertEqual(d["total"], sum(len(ep["decisiones"]) for ep in a.historial))
        self.assertEqual(d["greedy"] + d["llm"] + d["llm_invalidas"] + d["azar"], d["total"])
        self.assertEqual((d["consultas_llm"]["consultas"], d["consultas_llm"]["tiempos_agotados"]), (2, 1))
        c = rep["configuraciones"]
        self.assertEqual(c["evaluadas"], len(registros))
        self.assertLessEqual(c["menor_costo"]["C"], c["mayor_costo"]["C"])
        self.assertEqual(c["menor_costo"]["C"], min(r["C"] for r in registros))
        self.assertEqual(rep["dijkstra_y_simulacion"]["mapeos"], sum(len(e.operaciones) for _, e, _ in evaluaciones))
        self.assertGreater(rep["recursos"]["nucleos_maquina"], 0)
        self.assertIn("cpu_total_s", rep["recursos"])
        texto = reporte_corrida.markdown(rep)
        for seccion in ("## Tiempo", "## Episodios", "## Decisiones del agente", "## Dijkstra y simulación",
                        "## Configuraciones", "### Configuración de menor costo", "### Configuración de mayor costo",
                        "## Recursos computacionales"):
            self.assertIn(seccion, texto)
        with tempfile.TemporaryDirectory() as tmp:
            reporte_corrida.escribir(tmp, rep)
            self.assertTrue((Path(tmp) / "reporte_corrida.md").exists())
            json.loads((Path(tmp) / "reporte_corrida.json").read_text())


class TestEnlaceRL(unittest.TestCase):
    """Con epsilon, el agente RL decide el paso (en lugar de la exploración al azar); el resto, Dijkstra greedy."""

    def agente(self, eleccion, **cambios):
        malla = Agente(None, BASE, parametros(malla=[5, 6])).malla()
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(5, 6).ids())}}
        return Agente(lambda c: evaluacion_falsa(malla, nops), BASE, parametros(malla=[5, 6], **cambios),
                      eleccion_rl=eleccion)

    def test_1_rl_decide_con_probabilidad_epsilon(self):
        """Con epsilon 0.5 hay pasos del RL y pasos de Dijkstra greedy, y el RL decide lo que devolvió."""
        llamadas = []

        def rl(actual, libres, pesos, camino, opciones, c_actual, contexto, parada=None):
            if parada:
                return 0  # en el punto de paro: parar
            self.assertEqual(set(opciones), set(libres))  # el C de cada vecino libre
            elegido = libres[-1]
            llamadas.append((actual, list(libres), elegido))
            return elegido

        agente = self.agente(rl, epsilon=0.5, semilla=3)
        camino = agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"rl", "explotacion"})
        pasos_rl = [i for i, d in enumerate(agente.decisiones) if d == "rl"]
        self.assertEqual(len(llamadas), len(pasos_rl))
        for paso, (actual, libres, elegido) in zip(pasos_rl, llamadas):
            self.assertEqual(camino[paso], elegido)
            self.assertEqual(camino[paso - 1], actual)
            self.assertIn(elegido, libres)
        self.assertEqual(agente.resultado()["rl"], len(pasos_rl))

    def test_2_con_epsilon_0_no_se_llama_al_rl(self):
        """Epsilon 0 es Greedy puro: el agente RL nunca se consulta."""
        def rl(*args):
            raise AssertionError("el agente RL no debía consultarse")

        agente = self.agente(rl, epsilon=0.0)
        agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"explotacion"})
        self.assertEqual(agente.extensiones, [])  # ni en el punto de paro

    def test_3_respuesta_invalida_cae_a_greedy(self):
        """Si el RL no devuelve un vecino libre, ese paso es Dijkstra greedy. Con epsilon 1 y un RL que
        siempre falla, el camino es el mismo que el de Greedy puro."""
        agente = self.agente(lambda *args, **kw: None, epsilon=1.0)
        camino = agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"rl_invalida"})
        # en el punto de paro, la respuesta inválida es parar
        self.assertEqual([(x["decision"], x["n"]) for x in agente.extensiones], [("rl_invalida", 0)])
        self.assertEqual(agente.resultado()["rl_invalidas"], len(camino) - 1)
        self.assertEqual(camino, self.agente(None, epsilon=0.0).buscar())

    def test_4_sin_rl_la_rama_es_exploracion(self):
        """Sin agente RL, la rama de epsilon sigue siendo la exploración al azar."""
        agente = self.agente(None, epsilon=1.0)
        agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"exploracion"})


class TestEleccionLLM(unittest.TestCase):
    """El adaptador del agente RL (un LLM por CLI): arma el prompt del paso y lee el PE de la respuesta."""

    LIBRES = ["PE01", "PE10"]
    PESOS = {"PE01": PesoPE("PE01", "E", 1, 0, 1, 1.0), "PE10": PesoPE("PE10", "M", 12, 0, 1, 12.0)}
    OPCIONES = {"PE01": {"C": 0.6881, "cumple_tipos": False}, "PE10": {"C": 0.8512, "cumple_tipos": True}}
    ARGS = ("PE00", LIBRES, PESOS, ["PE00"], OPCIONES, 10.0)

    def test_1_prompt_con_los_vecinos_libres(self):
        """El prompt pide minimizar C y da, por vecino, su tipo, su área w_tipo y el C del camino con él;
        ya no habla de "peso"."""
        prompt = construir_prompt(*self.ARGS)
        self.assertIn("PE01", prompt)
        self.assertIn("PE10", prompt)
        self.assertIn("Responde solo", prompt)
        self.assertIn("PE00", prompt)
        self.assertIn("minimizar el costo C", prompt)
        self.assertIn("C del camino si se agrega 0.6881 (no cumple los tipos de PE)", prompt)
        self.assertIn("área w_tipo 12, C del camino si se agrega 0.8512 (cumple los tipos de PE)", prompt)
        self.assertIn("su C es 10", prompt)
        self.assertNotIn("peso", prompt.lower())

    def test_2_solo_acepta_el_identificador(self):
        """La respuesta vale solo si es exactamente el identificador de un vecino libre (se toleran
        espacios, comillas invertidas y un punto final). Con más texto no se adivina: es None (el paso
        cae a greedy como rl_invalida); así "descarto PE10, elijo PE01" no se toma como PE10."""
        for respuesta in ("PE01", "PE01\n", "  PE01  ", "PE01.", "`PE01`", "``PE01``\n"):
            self.assertEqual(interpretar(respuesta, self.LIBRES), "PE01", respuesta)
        for respuesta in ("descarto PE10, elijo PE01", "Elijo PE01.", "PE00 ya está visitado; PE10",
                          "PE01 PE10", "PE01, PE10", "no sé", "", None, "PE99", "PE00", "PE011", "pe01"):
            self.assertIsNone(interpretar(respuesta, self.LIBRES), respuesta)

    def test_3_ejecuta_el_comando_con_el_prompt(self):
        eleccion = EleccionLLM(["claude", "-p", "{prompt}"])
        with mock.patch("dijkstra.rl_llm.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout="PE10\n", stderr="")) as run:
            elegido = eleccion(*self.ARGS)
        self.assertEqual(elegido, "PE10")
        argumentos = run.call_args.args[0]
        self.assertEqual(argumentos[:2], ["claude", "-p"])
        self.assertIn("PE01", argumentos[2])
        self.assertNotIn("{prompt}", argumentos)

    def test_4_fallas_del_comando(self):
        """Sin el comando o con código de salida distinto de 0 falla; el tiempo agotado es una respuesta vacía."""
        eleccion = EleccionLLM(["claude", "-p", "{prompt}"])
        with mock.patch("dijkstra.rl_llm.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "no se encontró"):
                eleccion(*self.ARGS)
        with mock.patch("dijkstra.rl_llm.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="no autenticado")):
            with self.assertRaisesRegex(RuntimeError, "no autenticado"):
                eleccion(*self.ARGS)
        with mock.patch("dijkstra.rl_llm.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertIsNone(eleccion(*self.ARGS))
        with self.assertRaisesRegex(ValueError, "{prompt}"):
            EleccionLLM(["claude", "-p"])


class TestAgenteConEvaluador(_ConOperaciones):
    """El agente con el evaluador real (Dijkstra + simulador)."""

    def test_1_camino_con_el_evaluador_real(self):
        """La malla 3x3 simulada da NOPs por PE, pesos para los 9 PEs y un camino desde PE00."""
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas)
        agente = Agente(evaluador.evaluar, BASE, parametros())
        camino = agente.buscar()
        self.assertEqual(agente.evaluacion.estado, "correcta")
        self.assertEqual(len(agente.pesos), 9)
        operacion = agente.evaluacion.operaciones[self.pequena]
        self.assertEqual(sum(operacion.nops_por_pe.values()), operacion.nops)
        self.assertEqual(camino[0], "PE00")
        self.assertEqual(len(camino), len(set(camino)))
        self.assertGreater(agente.valuar_camino()["C"], 0)

    def test_2_en_paralelo_da_lo_mismo(self):
        """Simular cada operación en su propio proceso da el mismo costo y los mismos NOPs por PE."""
        agente = Agente(None, BASE, parametros())
        configuracion = agente.configuracion()
        serie = Evaluador(list(self.nombres), carpeta_programas=self.programas).evaluar(configuracion)
        paralelo = medir_en_paralelo(configuracion, list(self.nombres), self.programas, RUTA_COSTOS, procesos=2)
        self.assertEqual(paralelo.estado, serie.estado)
        self.assertAlmostEqual(paralelo.costo, serie.costo)
        for op in self.nombres:
            self.assertEqual(paralelo.operaciones[op].nops_por_pe, serie.operaciones[op].nops_por_pe)


class TestCompilarSkill02(unittest.TestCase):
    """El ciclo corre los skills 02-03 solo cuando falta algo o operaciones.c cambió."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()
        raiz = Path(self.directorio.name)
        shutil.copytree(FUENTE.parent, raiz / "operaciones")
        self.fuente = raiz / "operaciones" / "operaciones.c"
        self.programas = raiz / "programas"

    def tearDown(self):
        self.directorio.cleanup()

    def test_1_genera_lo_que_falta_una_vez(self):
        """Sin archivos se compila cada operación, y la segunda vez no hace falta nada."""
        operaciones = SimuladorCGRA.operaciones(self.fuente)
        self.assertEqual(preparar(operaciones, self.programas, self.fuente), operaciones)
        for operacion in operaciones:
            carpeta = self.programas / operacion
            nombres = {p.name for p in carpeta.iterdir()}
            self.assertLessEqual({"memoria.bin", "operaciones.ll", *grafos(operacion)}, nombres)  # skills 02-03
            self.assertNotIn("operaciones.out", nombres)
            self.assertIn('source_filename = "../../operaciones/operaciones.c"',
                          (carpeta / "operaciones.ll").read_text(encoding="utf-8"))
        self.assertEqual(preparar(operaciones, self.programas, self.fuente), [])

    def test_2_recompila_si_cambia_el_codigo(self):
        """Un operaciones.c más nuevo que las salidas, o forzar, vuelve a generar."""
        una = SimuladorCGRA.operaciones(self.fuente)[:1]
        preparar(una, self.programas, self.fuente)
        memoria = self.programas / una[0] / "memoria.bin"
        viejo = self.fuente.stat().st_mtime - 10
        os.utime(memoria, (viejo, viejo))
        self.assertEqual(preparar(una, self.programas, self.fuente), una)
        self.assertEqual(preparar(una, self.programas, self.fuente), [])
        self.assertEqual(preparar(una, self.programas, self.fuente, forzar=True), una)

    def test_3_errores_claros(self):
        """Una operación desconocida o un compilador que no existe dan ErrorCompilacion."""
        with self.assertRaises(ErrorCompilacion):
            preparar(["no_existe"], self.programas, self.fuente)
        with self.assertRaisesRegex(ErrorCompilacion, "No se encontró"):
            preparar(SimuladorCGRA.operaciones(self.fuente)[:1], self.programas, self.fuente,
                     clang="clang-no-existe")
        with self.assertRaisesRegex(ErrorCompilacion, "No se encontró"):
            preparar(SimuladorCGRA.operaciones(self.fuente)[:1], self.programas, self.fuente, opt="opt-no-existe")
        self.assertFalse((self.programas / "no_existe" / "operaciones.out").exists())


if __name__ == "__main__":
    unittest.main()
