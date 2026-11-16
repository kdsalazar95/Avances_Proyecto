"""Pruebas del agente Greedy (dijkstra_greedy/dijkstra/agente.py): camino sin ciclos por los PEs
de una malla, siempre al vecino sin visitar de menor peso w_tipo · (1 + NOPs/ciclos)."""

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

from dijkstra.agente import Agente, Parametros, PesoPE, medir_en_paralelo
from dijkstra.rl_copilot import EleccionCopilot, construir_prompt, interpretar
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
    """Greedy camina por los PEs: siempre al vecino sin visitar de menor peso, sin repetir ninguno."""

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
        """Empieza en PE00; cada paso va a un vecino sin visitar y de menor peso entre los libres;
        ningún PE se repite y termina cuando el último no tiene vecinos sin visitar."""
        malla = Agente(None, BASE, parametros(malla=[4, 5])).malla()
        nops = {"op": {pe: (i * 7) % 5 for i, pe in enumerate(Malla(4, 5).ids())}}
        agente = Agente(lambda c: evaluacion_falsa(malla, nops), BASE, parametros(malla=[4, 5]))
        camino = agente.buscar()
        pesos = agente.pesos
        self.assertEqual(camino[0], "PE00")
        self.assertEqual(len(camino), len(set(camino)))
        for i, (actual, siguiente) in enumerate(zip(camino, camino[1:])):
            libres = [v for v in Agente.vecinos(actual, 4, 5) if v not in camino[:i + 1]]
            self.assertIn(siguiente, libres)
            self.assertEqual(pesos[siguiente].peso, min(pesos[v].peso for v in libres))
        self.assertEqual([v for v in Agente.vecinos(camino[-1], 4, 5) if v not in camino], [])

    def test_5_empates_en_orden_fila_por_fila(self):
        """Con todos los pesos iguales por tipo, el empate se resuelve por el primero fila por fila."""
        agente = Agente(lambda c: evaluacion_falsa(MALLA_3X3, {"op": nops_iguales(MALLA_3X3, 0)}), BASE,
                        parametros())
        camino = agente.buscar()
        # PE00 (M) → PE01 (E, 1) → PE11 (V, 2; PE02 es M, 12) → PE10 (M) ... sin repetir.
        self.assertEqual(camino[:3], ["PE00", "PE01", "PE11"])
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
        """Con epsilon, cada paso explora (un vecino libre al azar) o explota (el de menor peso); el camino
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
                self.assertEqual(agente.pesos[siguiente].peso, min(agente.pesos[v].peso for v in libres))
        _, otro = correr(epsilon=1.0, semilla=3)
        self.assertNotEqual(otro, camino_greedy)
        r = agente.resultado()
        self.assertEqual(r["exploraciones"], agente.decisiones.count("exploracion"))
        self.assertEqual([p["decision"] for p in r["camino"]], agente.decisiones)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            correr(epsilon=1.5)


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

        def rl(actual, libres, pesos, camino):
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

    def test_3_respuesta_invalida_cae_a_greedy(self):
        """Si el RL no devuelve un vecino libre, ese paso es Dijkstra greedy. Con epsilon 1 y un RL que
        siempre falla, el camino es el mismo que el de Greedy puro."""
        agente = self.agente(lambda *args: None, epsilon=1.0)
        camino = agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"rl_invalida"})
        self.assertEqual(agente.resultado()["rl_invalidas"], len(camino) - 1)
        self.assertEqual(camino, self.agente(None, epsilon=0.0).buscar())

    def test_4_sin_rl_la_rama_es_exploracion(self):
        """Sin agente RL, la rama de epsilon sigue siendo la exploración al azar."""
        agente = self.agente(None, epsilon=1.0)
        agente.buscar()
        self.assertEqual(set(agente.decisiones[1:]), {"exploracion"})


class TestEleccionCopilot(unittest.TestCase):
    """El adaptador del agente RL: arma el prompt del paso y lee el PE de la respuesta."""

    LIBRES = ["PE01", "PE10"]
    PESOS = {"PE01": PesoPE("PE01", "E", 1, 0, 1, 1.0), "PE10": PesoPE("PE10", "M", 12, 0, 1, 12.0)}

    def test_1_prompt_con_los_vecinos_libres(self):
        prompt = construir_prompt("PE00", self.LIBRES, self.PESOS, ["PE00"])
        self.assertIn("PE01", prompt)
        self.assertIn("PE10", prompt)
        self.assertIn("Responde solo", prompt)
        self.assertIn("PE00", prompt)

    def test_2_interpreta_el_primer_vecino_libre(self):
        self.assertEqual(interpretar("Elijo PE01.\n", self.LIBRES), "PE01")
        self.assertEqual(interpretar("PE00 ya está visitado; PE10", self.LIBRES), "PE10")
        self.assertIsNone(interpretar("no sé", self.LIBRES))
        self.assertIsNone(interpretar("PE99", self.LIBRES))

    def test_3_ejecuta_el_comando_con_el_prompt(self):
        eleccion = EleccionCopilot(["copilot", "-p", "{prompt}", "--silent"])
        with mock.patch("dijkstra.rl_copilot.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, stdout="PE10\n", stderr="")) as run:
            elegido = eleccion("PE00", self.LIBRES, self.PESOS, ["PE00"])
        self.assertEqual(elegido, "PE10")
        argumentos = run.call_args.args[0]
        self.assertEqual(argumentos[:2], ["copilot", "-p"])
        self.assertIn("PE01", argumentos[2])
        self.assertNotIn("{prompt}", argumentos)

    def test_4_fallas_del_comando(self):
        """Sin el comando o con código de salida distinto de 0 falla; el tiempo agotado es una respuesta vacía."""
        eleccion = EleccionCopilot(["copilot", "-p", "{prompt}"])
        with mock.patch("dijkstra.rl_copilot.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "no se encontró"):
                eleccion("PE00", self.LIBRES, self.PESOS, ["PE00"])
        with mock.patch("dijkstra.rl_copilot.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="no autenticado")):
            with self.assertRaisesRegex(RuntimeError, "no autenticado"):
                eleccion("PE00", self.LIBRES, self.PESOS, ["PE00"])
        with mock.patch("dijkstra.rl_copilot.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertIsNone(eleccion("PE00", self.LIBRES, self.PESOS, ["PE00"]))
        with self.assertRaisesRegex(ValueError, "{prompt}"):
            EleccionCopilot(["copilot", "-p"])


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
