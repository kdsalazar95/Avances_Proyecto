"""Integración A08: el agente propone cgra.json; el evaluador (skills + simulador) lo mide."""

import sys
import tempfile
import unittest
from pathlib import Path

from rl import estados
from rl.__main__ import main as cli
from rl.agente.ciclo import Ciclo, ErrorCiclo, iniciar
from rl.agente.politica import PoliticaComando, PoliticaGuion, PoliticaManual
from rl.comun import escribir_json, leer_json, leer_jsonl
from rl.herramientas.referencia import escribir_cgra, malla_filas

from .utilidades import CompiladorCaido, corrida, editar_cgra


class TestCiclo(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.acciones = self.raiz / "acciones"
        # 'M V M' x 3, x 4 y x 2: memoria en los bordes oeste y este.
        for nombre, computo in (("V3", ["V"] * 3), ("V4", ["V"] * 4), ("V2", ["V"] * 2)):
            escribir_cgra(self.acciones / nombre, malla_filas(computo), id_candidata=nombre)

    def tearDown(self):
        self._tmp.cleanup()

    def _iniciar(self, **cambios):
        config = self.raiz / "corrida.json"
        escribir_json(config, corrida(**cambios))
        return iniciar(config, self.raiz / "corridas")

    def test_ciclo_completo_con_paciencia(self):
        """Mejora, empate, configuración que los skills no mapean y acción inválida; para por paciencia."""
        ciclo = self._iniciar(limites={"max_iteraciones": 20, "paciencia": 3, "t_max_simulacion_s": 60})
        self.assertEqual(ciclo.memoria.entradas[0]["estado_evaluacion"], estados.CORRECTA)
        otra = self.raiz / "V3-otra"    # mismo hardware, otro cgra.json (otra descripción): empata
        escribir_cgra(otra, malla_filas(["V"] * 3))
        editar_cgra(otra, lambda d: d.update(descripcion="otra"))
        no_mapea = self.raiz / "3x4"    # válida para A03, pero los skills (sustituto) no la saben mapear
        escribir_cgra(no_mapea, ["M E E M", "M V V M", "M E E M"])
        vacia = self.raiz / "vacia"
        vacia.mkdir()
        guion = [self.acciones / "V3", otra, no_mapea, vacia, self.acciones / "V4"]
        self.assertEqual(ciclo.correr(PoliticaGuion(guion)), "terminada: paciencia")
        h = leer_jsonl(ciclo.carpeta / "historial.jsonl")
        self.assertEqual([e["estado_evaluacion"] for e in h],
                         [estados.CORRECTA, estados.CORRECTA, estados.CORRECTA, estados.FALLO_EVALUADOR,
                          estados.ACCION_INVALIDA])
        optimas = leer_json(ciclo.carpeta / "optimas.json")
        self.assertEqual([o["id_candidata"] for o in optimas["optimas"]], ["it-0001", "it-0002"])
        self.assertAlmostEqual(optimas["C_min"], 78 / 76 + 73 / 129)  # C_A de V3: 6·12 + 3·2
        self.assertEqual(h[3]["recompensa"], -10.0)
        self.assertEqual(h[4]["errores"][0]["skill"], "A03")
        resumen = leer_json(ciclo.carpeta / "resumen.json")
        self.assertEqual((resumen["porcentaje_validas"], resumen["porcentaje_correctas"]), (75.0, 50.0))
        # La acción del agente queda sin programas; los PE*.txt están en la evaluación.
        self.assertFalse((ciclo.carpeta / "candidatas" / "it-0001" / "pe_instructions").exists())
        self.assertTrue((ciclo.carpeta / "eval" / "it-0001" / "pe_instructions" / "PE00.txt").is_file())
        with self.assertRaises(PermissionError):
            (ciclo.carpeta / "candidatas" / "it-0001" / "cgra.json").write_text("x")

    def test_max_iteraciones_y_reanudar(self):
        ciclo = self._iniciar(limites={"max_iteraciones": 1, "paciencia": 5, "t_max_simulacion_s": 60})
        self.assertEqual(ciclo.correr(PoliticaGuion([self.acciones / "V4"])), "terminada: max_iteraciones")
        reanudado = Ciclo(ciclo.carpeta)
        self.assertEqual(reanudado.siguiente_id(), "it-0002")
        self.assertEqual(reanudado.normalizadores, ciclo.normalizadores)
        estado = reanudado.correr(PoliticaGuion([self.acciones / "V3"]), max_iteraciones=2)
        self.assertEqual(estado, "terminada: max_iteraciones")
        self.assertEqual(len(reanudado.memoria.entradas), 3)
        self.assertEqual(reanudado.memoria.por_id("it-0001")["metricas"]["ciclos"], 63)  # 'M V M' x 4
        self.assertEqual(reanudado.memoria.por_id("it-0001")["metricas"]["C_A"], 104)  # 8·12 + 4·2

    def test_error_de_infraestructura_no_cuenta(self):
        """Un error de infraestructura no penaliza al agente ni consume paciencia."""
        ciclo = self._iniciar(limites={"max_iteraciones": 2, "paciencia": 1, "t_max_simulacion_s": 60})
        ciclo.compilador = CompiladorCaido()
        estado = ciclo.correr(PoliticaGuion([self.acciones / "V3", self.acciones / "V4"]))
        self.assertEqual(estado, "terminada: max_iteraciones")
        h = ciclo.memoria.entradas[1:]
        self.assertEqual({e["estado_evaluacion"] for e in h}, {estados.ERROR_INFRAESTRUCTURA})
        self.assertEqual({e["recompensa"] for e in h}, {None})

    def test_politica_manual_y_cli(self):
        """La política manual deja el prompt y pausa; 'evaluar' con la CLI continúa la corrida."""
        ciclo = self._iniciar()
        self.assertEqual(ciclo.correr(PoliticaManual(ciclo.carpeta / "prompts")),
                         "en_pausa: esperando acción del agente")
        prompt = (ciclo.carpeta / "prompts" / "it-0001.md").read_text(encoding="utf-8")
        self.assertIn("Tu acción es la configuración de la CGRA", prompt)
        self.assertIn("No escribas PE*.txt", prompt)
        self.assertNotIn("hash_accion", prompt)
        # Rutas relativas (como se escriben en la terminal): el simulador corre en cgra/src y debe encontrarlas.
        import os
        codigo = cli(["evaluar", "--corrida", os.path.relpath(ciclo.carpeta),
                      "--candidata", os.path.relpath(self.acciones / "V2")])
        self.assertEqual(codigo, 0)
        h = leer_jsonl(ciclo.carpeta / "historial.jsonl")
        self.assertEqual(h[-1]["id_evaluacion"], "V2")    # id tomado de candidata.json
        self.assertEqual(h[-1]["estado_evaluacion"], estados.CORRECTA)

    def test_politica_comando(self):
        """Un LLM externo (aquí un script) recibe el prompt y escribe el cgra.json en dir_salida."""
        ciclo = self._iniciar(limites={"max_iteraciones": 1, "paciencia": 3, "t_max_simulacion_s": 60})
        script = self.raiz / "agente_falso.py"
        script.write_text(
            "import sys, shutil, json\n"
            "from pathlib import Path\n"
            "prompt, destino, origen = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])\n"
            "assert 'Qué entregar' in prompt.read_text()\n"
            "destino.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy(origen / 'cgra.json', destino / 'cgra.json')\n"
            "(destino/'candidata.json').write_text(json.dumps({'justificacion': 'menos área'}))\n",
            encoding="utf-8")
        politica = PoliticaComando([sys.executable, str(script), "{prompt_archivo}", "{dir_salida}",
                                    str(self.acciones / "V3")], ciclo.carpeta / "prompts")
        self.assertEqual(ciclo.correr(politica), "terminada: max_iteraciones")
        entrada = ciclo.memoria.por_id("it-0001")
        self.assertEqual(entrada["estado_evaluacion"], estados.CORRECTA)
        self.assertEqual(entrada["justificacion_agente"], "menos área")

    def test_referencia_que_no_se_puede_evaluar_detiene(self):
        """Si la referencia no resulta correcta no se inicia la optimización."""
        ref = self.raiz / "ref-3x4"
        escribir_cgra(ref, ["M E E M", "M V V M", "M E E M"])
        with self.assertRaises(ErrorCiclo):
            self._iniciar(referencia={"cgra": str(ref / "cgra.json")})
        resumen = leer_json(self.raiz / "corridas" / "prueba" / "resumen.json")
        self.assertIn("referencia no válida", resumen["estado"])

    def test_no_sobrescribe_corrida(self):
        self._iniciar()
        with self.assertRaises(ErrorCiclo):
            self._iniciar()


if __name__ == "__main__":
    unittest.main()
