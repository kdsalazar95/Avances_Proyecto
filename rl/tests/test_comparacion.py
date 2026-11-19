"""Comparación entre sistemas: normalizadores fijos, epsilon sobre el LLM, convergencia y barrido de epsilon."""

import csv
import random
import tempfile
import unittest
from pathlib import Path

from rl import estados
from rl.__main__ import main as cli
from rl.agente.ciclo import ErrorCiclo, iniciar
from rl.agente.politica import PoliticaEpsilon, PoliticaGuion, configuracion_al_azar
from rl.comun import ESQUEMA_CGRA, escribir_json, leer_json, leer_jsonl
from rl.corrida import ErrorCorrida
from rl.entorno.validacion_accion import validar_accion
from rl.herramientas.barrido import barrido, leer_epsilons, nombre
from rl.herramientas.convergencia import COLUMNAS, escribir
from rl.herramientas.referencia import escribir_cgra, malla_filas

from .utilidades import corrida


class _Base(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.acciones = self.raiz / "acciones"
        for nombre_, computo in (("V3", ["V"] * 3), ("V4", ["V"] * 4), ("V2", ["V"] * 2), ("E3", ["E"] * 3)):
            escribir_cgra(self.acciones / nombre_, malla_filas(computo), id_candidata=nombre_)

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, nombre_="corrida.json", **cambios):
        ruta = self.raiz / nombre_
        escribir_json(ruta, corrida(**cambios))
        return ruta

    def guion(self, *nombres):
        return PoliticaGuion([self.acciones / n for n in nombres])


class TestNormalizadoresFijos(_Base):

    def test_1_la_referencia_se_evalua_con_los_normalizadores_fijos(self):
        """Con 'normalizadores' en la corrida, la referencia no mide los suyos: se evalúa con los fijos, su
        recompensa es 0 y no cuenta para la paciencia. Si los fijos son los que mediría, C_ref = K_A + K_L = 2."""
        medida = iniciar(self.config(), self.raiz / "corridas")
        medidos = leer_json(medida.carpeta / "normalizacion.json")
        fijos = self.raiz / "normalizacion.json"
        escribir_json(fijos, {"formula_latencia": "ciclos", "C_A_ref": medidos["C_A_ref"],
                              "operaciones": {"reduccion": {"C_L_ref": medidos["C_L_ref"]}}})
        ciclo = iniciar(self.config("fijos.json", id="fijos", normalizadores=str(fijos)), self.raiz / "corridas")
        ref = ciclo.memoria.entradas[0]
        self.assertTrue(ref["es_referencia"])
        self.assertEqual((ref["estado_evaluacion"], ref["recompensa"]), (estados.CORRECTA, 0.0))
        self.assertAlmostEqual(ref["costo_total"], 2.0)
        self.assertTrue(ciclo.normalizadores["fijos"])
        self.assertEqual(ciclo.normalizadores["C_L_ref"], medidos["C_L_ref"])
        # las candidatas cuestan lo mismo que con los normalizadores medidos
        ciclo.correr(self.guion("V3"), max_iteraciones=1)
        medida.correr(self.guion("V3"), max_iteraciones=1)
        self.assertAlmostEqual(ciclo.memoria.entradas[1]["costo_total"], medida.memoria.entradas[1]["costo_total"])

    def test_2_otra_formula_o_faltan_operaciones(self):
        fijos = self.raiz / "n.json"
        escribir_json(fijos, {"formula_latencia": "ciclos_x_nops", "C_A_ref": 76,
                              "operaciones": {"reduccion": {"C_L_ref": 100}}})
        with self.assertRaises(ErrorCorrida):
            iniciar(self.config(normalizadores=str(fijos)), self.raiz / "corridas")
        escribir_json(fijos, {"formula_latencia": "ciclos", "C_A_ref": 76, "operaciones": {"matmul": {"C_L_ref": 1}}})
        with self.assertRaises(ErrorCorrida):
            iniciar(self.config(normalizadores=str(fijos)), self.raiz / "corridas")
        self.assertFalse((self.raiz / "corridas" / "prueba").exists())

    def test_3_la_configuracion_de_la_comparacion(self):
        """rl/config/comparacion.json: la 32x32 de cgra.json (fila 0 E), normalizadores fijos, paciencia 3."""
        from rl.corrida import cargar_corrida, normalizadores_fijos
        c = cargar_corrida(Path(__file__).resolve().parents[1] / "config" / "comparacion.json")
        malla = c["referencia"]["generar"]["malla"]
        self.assertEqual((len(malla), len(malla[0].split())), (32, 32))
        self.assertEqual(malla[0].split()[:3] + malla[1].split()[:2], ["M", "E", "E", "M", "V"])
        self.assertTrue(all(f.split()[0] == f.split()[-1] == "M" for f in malla))
        self.assertEqual((c["limites"]["max_iteraciones"], c["limites"]["paciencia"]), (10, 3))
        self.assertNotIn("requiere_escalar_y_vectorial", c["restricciones"])
        self.assertEqual(normalizadores_fijos(c)["operaciones"]["matmul"]["C_L_ref"], 904.8888888888889)
        self.assertEqual(c["politica"]["tipo"], "comando")


class TestEpsilon(_Base):

    def test_1_epsilon_0_es_el_llm_y_1_es_al_azar(self):
        ciclo = iniciar(self.config(), self.raiz / "corridas")
        p = PoliticaEpsilon(self.guion("V3", "V4"), 0.0, 0, ESQUEMA_CGRA)
        ciclo.correr(p, max_iteraciones=2)
        self.assertEqual([e["decidio"] for e in ciclo.memoria.entradas], ["referencia", "llm", "llm"])

        azar = iniciar(self.config(id="azar", epsilon=1.0, semilla=7), self.raiz / "corridas")
        guion = self.guion("V3")
        azar.correr(PoliticaEpsilon(guion, 1.0, 7, ESQUEMA_CGRA), max_iteraciones=3)
        h = azar.memoria.entradas[1:]
        self.assertEqual({e["decidio"] for e in h}, {"exploracion"})
        self.assertEqual(len(guion.paquetes), 1)  # el LLM no se consultó
        for e in h:  # las configuraciones al azar cumplen las restricciones (A03)
            self.assertNotEqual(e["estado_evaluacion"], estados.ACCION_INVALIDA, e["errores"][:1])

    def test_2_mezcla_y_misma_semilla_mismas_decisiones(self):
        def decisiones(semilla, carpeta):
            c = iniciar(self.config(id=carpeta), self.raiz / "corridas")
            c.correr(PoliticaEpsilon(self.guion("V3", "V4", "V2", "E3"), 0.5, semilla, ESQUEMA_CGRA),
                     max_iteraciones=4)
            return [e["decidio"] for e in c.memoria.entradas[1:]], [e["malla"] for e in c.memoria.entradas[1:]]

        a = decisiones(3, "a")
        self.assertEqual(a, decisiones(3, "b"))
        self.assertEqual(set(a[0]), {"llm", "exploracion"})

    def test_3_configuracion_al_azar_valida(self):
        ciclo = iniciar(self.config(restricciones={"memoria_solo_en_bordes_oeste_este": True,
                                                   "requiere_memoria_y_computo": True,
                                                   "filas": {"minimo": 1, "maximo": 32},
                                                   "columnas": {"minimo": 3, "maximo": 32}}),
                        self.raiz / "corridas")
        azar = random.Random(1)
        tamanos = set()
        for k in range(30):
            malla = configuracion_al_azar(azar, ciclo.contexto["restricciones"])
            tamanos.add((len(malla), len(malla[0].split())))
            destino = escribir_cgra(self.raiz / "azar" / f"c{k}", malla).parent
            self.assertTrue(validar_accion(destino, ciclo.contexto, f"c{k}")["paquete_valido"], malla)
            self.assertTrue(all(f.split()[0] == f.split()[-1] == "M" for f in malla))
            self.assertTrue({x for f in malla for x in f.split()[1:-1]} <= {"E", "V"})
        self.assertGreater(len(tamanos), 20)


class TestConvergencia(_Base):

    def test_1_datos_de_convergencia(self):
        """convergencia.csv: una fila por evaluación (n = 0 la referencia, tiempo 0), C_best no sube y el
        tiempo acumulado no baja; convergencia.json: la evaluación en que se alcanzó el mejor C y los hitos."""
        ciclo = iniciar(self.config(limites={"max_iteraciones": 10, "paciencia": 2, "t_max_simulacion_s": 60}),
                        self.raiz / "corridas")
        estado = ciclo.correr(PoliticaEpsilon(self.guion("V4", "V3", "V2", "E3", "V4"), 0.0, 0, ESQUEMA_CGRA))
        self.assertEqual(estado, "terminada: paciencia")
        with (ciclo.carpeta / "convergencia.csv").open(encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        self.assertEqual(tuple(filas[0]), COLUMNAS)
        self.assertEqual(len(filas), len(ciclo.memoria.entradas))
        self.assertEqual((filas[0]["n"], filas[0]["decidio"], float(filas[0]["tiempo_acumulado_s"])),
                         ("0", "referencia", 0.0))
        c_best = [float(f["C_best"]) for f in filas]
        self.assertEqual(c_best, sorted(c_best, reverse=True))
        tiempos = [float(f["tiempo_acumulado_s"]) for f in filas]
        self.assertEqual(tiempos, sorted(tiempos))
        datos = leer_json(ciclo.carpeta / "convergencia.json")
        self.assertEqual(datos["motivo_fin"], "terminada: paciencia")
        self.assertEqual(datos["evaluaciones"], len(filas) - 1)
        self.assertAlmostEqual(datos["C_best"], min(c_best))
        n_conv = datos["convergencia"]["evaluacion"]
        self.assertAlmostEqual(float(filas[n_conv]["C"]), datos["C_best"])
        self.assertTrue(all(float(f["C_best"]) > datos["C_best"] for f in filas[:n_conv]))
        self.assertEqual(datos["hitos"]["100"]["evaluacion"], n_conv)
        self.assertLessEqual(datos["hitos"]["50"]["evaluacion"], datos["hitos"]["90"]["evaluacion"])
        self.assertAlmostEqual(datos["mejora_relativa"], (datos["C_ref"] - datos["C_best"]) / datos["C_ref"], places=5)
        # se puede rehacer desde el historial
        self.assertEqual(escribir(ciclo.carpeta), datos)


class TestBarrido(_Base):

    def test_1_una_corrida_por_epsilon_con_la_misma_referencia(self):
        self.assertEqual(leer_epsilons(["0", "10", "20", "80", "90", "100"]), [0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        self.assertEqual(nombre("s1", 0.1), "s1-eps010")
        config = self.config(id="s1", limites={"max_iteraciones": 3, "paciencia": 3, "t_max_simulacion_s": 60})
        corridas = self.raiz / "corridas"
        carpetas, resumenes = barrido(config, [0.0, 1.0], corridas,
                                      politica=lambda c: PoliticaEpsilon(self.guion("V3", "V4", "V2"),
                                                                         c.corrida["epsilon"], 0, ESQUEMA_CGRA))
        self.assertEqual([c.name for c in carpetas], ["s1-eps000", "s1-eps100"])
        ref0, ref1 = (leer_jsonl(c / "historial.jsonl")[0] for c in carpetas)
        self.assertNotIn("reusada_de", ref0)
        self.assertIn("s1-eps000", ref1["reusada_de"])  # la referencia se evaluó una sola vez
        self.assertEqual(ref0["costo_total"], ref1["costo_total"])
        self.assertEqual([r["epsilon"] for r in resumenes], [0.0, 1.0])
        salida = corridas / "s1-comparacion"
        with (salida / "comparacion_epsilon.csv").open(encoding="utf-8") as f:
            tabla = list(csv.DictReader(f))
        self.assertEqual([t["epsilon"] for t in tabla], ["0.0", "1.0"])
        self.assertEqual(tabla[1]["fraccion_exploracion"], "1.0")
        with (salida / "curvas_epsilon.csv").open(encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), sum(len(leer_jsonl(c / "historial.jsonl")) for c in carpetas))
        # reanudar: no repite nada
        barrido(config, [0.0, 1.0], corridas, politica=lambda c: PoliticaGuion([]))
        self.assertEqual(len(leer_jsonl(carpetas[0] / "historial.jsonl")), 4)
        # la CLI junta corridas
        self.assertEqual(cli(["comparar", str(carpetas[0]), str(carpetas[1]), "--salida", str(self.raiz / "c")]), 0)
        self.assertTrue((self.raiz / "c" / "comparacion_epsilon.csv").is_file())

    def test_2_politica_manual_no_sirve_para_el_barrido(self):
        with self.assertRaises(ErrorCiclo):
            barrido(self.config(), [0.0, 0.5], self.raiz / "corridas")
        self.assertFalse((self.raiz / "corridas").exists())


if __name__ == "__main__":
    unittest.main()
