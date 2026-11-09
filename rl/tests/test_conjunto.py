"""Una configuración evaluada con todas las operaciones de la corrida (la CGRA no sabe cuál corre)."""

import tempfile
import unittest
from pathlib import Path

from rl import estados
from rl.agente.ciclo import iniciar
from rl.comun import escribir_json
from rl.herramientas.referencia import escribir_cgra, malla_filas
from rl.evaluador.compilador import CompiladorReduccion
from rl.evaluador.conjunto import evaluar_conjunto

from .utilidades import Temporal, corrida


class CompiladorQueFallaEn(CompiladorReduccion):
    """Compila con el sustituto, salvo en las carpetas de evaluación de ``operacion`` (los skills reportan)."""

    def __init__(self, operacion, infraestructura=False):
        self.operacion = operacion
        self.infraestructura = infraestructura

    def compilar(self, ruta_cgra, instancia, dir_salida, errores_previos=None):
        if f"/{self.operacion}/" in str(dir_salida):
            return {"ok": False, "infraestructura": self.infraestructura, "detalle": "no se pudo mapear"}
        return super().compilar(ruta_cgra, instancia, dir_salida, errores_previos)


class TestEvaluarConjunto(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t = Temporal()
        # Dos "operaciones" con la misma instancia de reducción: basta para probar la agregación.
        cls.instancias = {"reduccion": cls.t.instancia, "otra": cls.t.instancia}
        cls.c = corrida()
        cls.ref = cls.t.config(("E", "V", "E"), True, "ref")
        cls.r_ref = evaluar_conjunto(cls.ref, "referencia", cls.instancias, cls.c, None, None,
                                     cls.t.raiz / "eval" / "referencia", CompiladorReduccion())

    @classmethod
    def tearDownClass(cls):
        cls.t._tmp.cleanup()

    def test_referencia_por_operacion(self):
        """La referencia es correcta en todas, cuesta K_A + K_L y deja un C_L,ref por operación."""
        r = self.r_ref
        self.assertEqual(r["estado_evaluacion"], estados.CORRECTA, r["errores"])
        self.assertAlmostEqual(r["costo_total"], 2.0)
        norm = r["parametros"]["normalizadores"]
        self.assertEqual(set(norm["operaciones"]), {"reduccion", "otra"})
        self.assertEqual(r["metricas"]["ciclos"], sum(d["ciclos"] for d in r["por_operacion"].values()))
        self.assertTrue((self.t.raiz / "eval" / "referencia" / "otra" / "pe_instructions" / "PE00.txt").is_file())

    def test_costo_es_el_promedio(self):
        """Correcta en todas: el costo es el promedio de los costos de cada operación."""
        norm = self.r_ref["parametros"]["normalizadores"]
        accion = self.t.config(("V", "V"), True, "vv")
        r = evaluar_conjunto(accion, "it-0001", self.instancias, self.c, norm, 2.0,
                             self.t.raiz / "eval" / "it-0001", CompiladorReduccion())
        self.assertEqual(r["estado_evaluacion"], estados.CORRECTA, r["errores"])
        costos = [d["costo"] for d in r["por_operacion"].values()]
        self.assertAlmostEqual(r["costo_total"], sum(costos) / len(costos))
        self.assertAlmostEqual(r["recompensa"], 2.0 - r["costo_total"])

    def test_si_falla_una_falla_la_configuracion(self):
        """Basta que una operación falle para que la configuración no sea correcta y se penalice."""
        norm = self.r_ref["parametros"]["normalizadores"]
        accion = self.t.config(("V", "V"), True, "falla")
        r = evaluar_conjunto(accion, "it-0002", self.instancias, self.c, norm, 2.0,
                             self.t.raiz / "eval" / "it-0002", CompiladorQueFallaEn("otra"))
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_EVALUADOR)
        self.assertEqual(r["por_operacion"]["reduccion"]["estado"], estados.CORRECTA)
        self.assertEqual(r["costo_total"], 10.0)
        self.assertEqual(r["recompensa"], -10.0)
        self.assertTrue(all(e["operacion"] == "otra" for e in r["errores"]))

    def test_infraestructura_tiene_prioridad(self):
        """Un error de infraestructura en una operación no se le cobra al agente."""
        norm = self.r_ref["parametros"]["normalizadores"]
        accion = self.t.config(("V", "V"), True, "infra")
        r = evaluar_conjunto(accion, "it-0003", self.instancias, self.c, norm, 2.0,
                             self.t.raiz / "eval" / "it-0003", CompiladorQueFallaEn("otra", infraestructura=True))
        self.assertEqual(r["estado_evaluacion"], estados.ERROR_INFRAESTRUCTURA)
        self.assertIsNone(r["recompensa"])


class TestCorridaConjunto(unittest.TestCase):

    def test_iniciar_y_evaluar_con_operaciones(self):
        """Con 'operaciones' en corrida.json se prepara una instancia por operación y la referencia las corre todas."""
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            datos = corrida(operaciones=["reduccion"])
            del datos["operacion"]
            escribir_json(raiz / "corrida.json", datos)
            ciclo = iniciar(raiz / "corrida.json", raiz / "corridas")
            self.assertTrue((ciclo.carpeta / "instancia" / "reduccion" / "memoria.bin").is_file())
            self.assertIn("operaciones", ciclo.normalizadores)
            self.assertEqual(ciclo.contexto["operaciones"], ["reduccion"])
            accion = raiz / "accion"
            escribir_cgra(accion, malla_filas(["V", "V"]))
            r = ciclo.evaluar_candidata(accion)
            self.assertEqual(r["estado_evaluacion"], estados.CORRECTA, r["errores"])
            self.assertIn("reduccion", r["por_operacion"])
            self.assertTrue(r["por_operacion"]["reduccion"]["nops_por_pe"], "NOPs de cada PE en el reporte")
            # Lote: varias configuraciones a la vez, registradas en orden y medidas contra el mejor previo.
            otras = []
            for k, filas in enumerate((["V", "V", "V"], ["E", "V"])):
                destino = raiz / f"lote{k}"
                escribir_cgra(destino, malla_filas(filas))
                otras.append(destino)
            mejor_previo = ciclo.memoria.mejor_costo()
            reportes = ciclo.evaluar_lote(otras, paralelo=2)
            self.assertEqual([x["id_evaluacion"] for x in reportes], ["it-0002", "it-0003"])
            for x in reportes:
                self.assertAlmostEqual(x["detalle_recompensa"]["C_best_prev"], mejor_previo)
            from rl.herramientas.informe import escribir_informe
            texto = escribir_informe(ciclo.carpeta).read_text(encoding="utf-8")
            for seccion in ("## 1. Óptima", "## 3. Frente de Pareto", "## 4. Todas las configuraciones"):
                self.assertIn(seccion, texto)
            self.assertIn("it-0003", texto)


if __name__ == "__main__":
    unittest.main()
