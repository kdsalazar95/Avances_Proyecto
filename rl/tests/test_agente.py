"""Pruebas del bloque agente: A03, A04, A05 y A07 (deterministas)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rl import estados
from rl.entorno.contexto import cargar_contexto
from rl.agente.memoria import Memoria
from rl.agente.optimas import actualizar_optimas
from rl.agente.politica import construir_prompt
from rl.entorno.validacion_accion import validar_accion
from rl.agente.recepcion import ErrorIntegracion, recibir_reporte
from rl.comun import ESQUEMA_CGRA, RAIZ_REPO
from rl.corrida import ErrorCorrida, cargar_corrida, validar_corrida
from rl.evaluador import reporte as e08

from .utilidades import Temporal, corrida, editar_cgra


def reporte_falso(id_, estado, costo, hash_="h", es_referencia=False):
    accion = {"id_candidata": id_, "directorio": f"d/{id_}", "configuracion": "c",
              "hashes": {"cgra.json": hash_}, "hash_accion": hash_}
    metricas = {"C_A": 1, "C_L": 1, "C_R": 0.0, "C_A_norm": 1, "C_L_norm": 1, "ciclos": 1, "nops": 0} \
        if estado == estados.CORRECTA else {}
    doc = e08.construir(id_, accion, estado, metricas, costo, 0.0 if estado == estados.CORRECTA else -10.0,
                        errores=[] if estado == estados.CORRECTA else [{"skill": "E03", "detalle": "x"}],
                        parametros={})
    doc["programas_generados"] = {"intentos": 1} if estado == estados.CORRECTA else None
    doc["es_referencia"] = es_referencia
    return doc


class TestCorrida(unittest.TestCase):
    def test_parametros_obligatorios(self):
        """corrida.json sin K o sin P_fallo se rechaza con un mensaje claro."""
        datos = corrida()
        del datos["K"]
        self.assertIn("falta 'K'", validar_corrida(datos))
        datos = corrida(penalizacion={})
        self.assertTrue(any("P_fallo" in e for e in validar_corrida(datos)))
        datos = corrida(compilador={"tipo": "comando"})
        self.assertTrue(any("compilador.comando" in e for e in validar_corrida(datos)))
        self.assertFalse(validar_corrida(corrida()))

    def test_config_de_ejemplo(self):
        """Las corridas del repositorio son válidas; la principal usa los Skills 01–09 con un LLM."""
        datos = cargar_corrida(RAIZ_REPO / "rl" / "config" / "corrida_reduccion.json")
        self.assertEqual(datos["pesos_area"], {"w1_MS": 4, "w2_MV": 8, "w3_CS": 1, "w4_CV": 2})
        self.assertEqual(datos["compilador"]["tipo"], "comando")
        self.assertTrue(datos["pendientes"])
        sin_llm = cargar_corrida(RAIZ_REPO / "rl" / "config" / "corrida_reduccion_sin_llm.json")
        self.assertEqual(sin_llm["compilador"]["tipo"], "reduccion_mcm")
        with self.assertRaises(ErrorCorrida):
            cargar_corrida(RAIZ_REPO / "no_existe.json")


class TestValidarAccion(unittest.TestCase):
    """A03: validación previa de la configuración (la acción del agente)."""

    @classmethod
    def setUpClass(cls):
        cls.t = Temporal()
        cls.contexto = cargar_contexto(corrida(), cls.t.instancia)

    @classmethod
    def tearDownClass(cls):
        cls.t.cerrar()

    def test_configuracion_valida(self):
        r = validar_accion(self.t.config(), self.contexto, "x")
        self.assertTrue(r["paquete_valido"], r["motivos"])

    def test_falta_cgra_json(self):
        vacia = self.t.carpeta("vacia")
        vacia.mkdir()
        self.assertEqual(validar_accion(vacia, self.contexto)["motivos"], ["falta cgra.json"])

    def test_mapa_que_no_coincide_con_el_tamano(self):
        """El mapa debe coincidir con filas x columnas (lo verifica FormatoCGRA)."""
        c = self.t.config()
        editar_cgra(c, lambda d: d.update(filas=4))
        motivos = validar_accion(c, self.contexto)["motivos"]
        self.assertTrue(any("cgra.json" in m and "filas" in m for m in motivos), motivos)

    def test_tipo_inventado_y_lista_cambiada(self):
        c = self.t.config()

        def cambio(d):
            d["tipos_pe"]["X"] = {"nombre": "x", "instrucciones": ["NOP"]}
            d["tipos_pe"]["M"]["instrucciones"].append("ADD")
        editar_cgra(c, cambio)
        motivos = " ".join(validar_accion(c, self.contexto)["motivos"])
        self.assertIn("'X' no existe", motivos)
        self.assertIn("cambia su lista", motivos)

    def test_limites_y_topologia(self):
        """Fuera del rango de la corrida, memoria fuera de los bordes o sin cómputo: inválida."""
        ctx = copy.deepcopy(self.contexto)
        ctx["restricciones"]["filas"] = {"minimo": 1, "maximo": 2}
        self.assertTrue(any("filas = 3" in m for m in validar_accion(self.t.config(), ctx)["motivos"]))
        c = self.t.config()
        editar_cgra(c, lambda d: d.update(malla=["M M M", "M V M", "M E M"]))
        self.assertTrue(any("PE(0,1) es M y accede a memoria" in m for m in validar_accion(c, self.contexto)["motivos"]))
        c = self.t.config()
        editar_cgra(c, lambda d: d.update(malla=["M M M"] * 3))
        self.assertIn("la malla no tiene ningún PE de cómputo", validar_accion(c, self.contexto)["motivos"])

    def test_memoria_en_bordes_y_computo_libre_en_el_interior(self):
        """Con memoria en los bordes, oeste y este deben ser todos M y el interior solo cómputo; el agente elige E, V o C en el interior."""
        r = self.contexto["restricciones"]
        todos = {letra: {"nombre": r["nombres_tipo"][letra], "instrucciones": ops}
                 for letra, ops in r["instrucciones_por_tipo"].items()}
        c = self.t.config()
        editar_cgra(c, lambda d: d.update(malla=["M C V M", "M E C M"], filas=2, columnas=4, tipos_pe=todos))
        self.assertEqual(validar_accion(c, self.contexto)["motivos"], [])
        editar_cgra(c, lambda d: d.update(malla=["E C M", "M V M"], filas=2, columnas=3))
        self.assertTrue(any("PE(0,0) es E: los bordes" in m for m in validar_accion(c, self.contexto)["motivos"]))
        editar_cgra(c, lambda d: d.update(malla=["M G M", "M V M"]))
        self.assertTrue(any("PE(0,1) es G y accede a memoria" in m for m in validar_accion(c, self.contexto)["motivos"]))
        editar_cgra(c, lambda d: d.update(malla=["M C"] * 2, columnas=2))
        self.assertTrue(any("al menos 3 columnas" in m for m in validar_accion(c, self.contexto)["motivos"]))

    def test_no_entrega_programas(self):
        """La acción no incluye PE*.txt: los generan los Skills 01–09 del evaluador."""
        c = self.t.config()
        (c / "pe_instructions").mkdir()
        motivos = validar_accion(c, self.contexto)["motivos"]
        self.assertTrue(any("Skills 01–09" in m for m in motivos))

    def test_contexto_usa_el_esquema_real(self):
        esquema = json.loads(Path(ESQUEMA_CGRA).read_text(encoding="utf-8"))
        self.assertEqual(set(self.contexto["restricciones"]["instrucciones_por_tipo"]), set(esquema["tipos_pe"]))
        self.assertEqual(self.contexto["parametros_instancia"], {"TAMANO_VECTOR": 100})
        self.assertIn("PE*.txt", self.contexto["entrega"]["no_entregar"])


class TestRecepcion(unittest.TestCase):
    """A04: el reporte debe identificar exactamente la acción enviada."""

    def test_reporte_coherente(self):
        r = reporte_falso("it-0001", estados.CORRECTA, 1.5)
        recibido = recibir_reporte(r, "it-0001")
        self.assertEqual((recibido["C"], recibido["e"]), (1.5, estados.CORRECTA))

    def test_incoherencias(self):
        """Otro id, una correcta sin métricas o hashes que no corresponden son errores de integración."""
        r = reporte_falso("it-0001", estados.CORRECTA, 1.5)
        with self.assertRaises(ErrorIntegracion):
            recibir_reporte(r, "it-0002")
        sin = copy.deepcopy(r)
        sin["metricas"]["ciclos"] = None
        with self.assertRaises(ErrorIntegracion):
            recibir_reporte(sin, "it-0001")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "cgra.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ErrorIntegracion):
                recibir_reporte(r, "it-0001", d)


class TestOptimas(unittest.TestCase):
    """A07: menor costo, mayor, empate, duplicado e incorrecta barata."""

    def test_reglas(self):
        tol = 1e-9
        o, c, mejora = actualizar_optimas([], reporte_falso("a", estados.CORRECTA, 1.0, "ha"), tol)
        self.assertEqual(([x["id_candidata"] for x in o], c, mejora), (["a"], 1.0, True))
        o, c, mejora = actualizar_optimas(o, reporte_falso("b", estados.CORRECTA, 1.0, "hb"), tol)
        self.assertEqual(([x["id_candidata"] for x in o], mejora), (["a", "b"], False))
        o2, _, _ = actualizar_optimas(o, reporte_falso("c", estados.CORRECTA, 1.0, "ha"), tol)
        self.assertEqual(len(o2), 2, "un paquete duplicado no se agrega dos veces")
        o3, _, _ = actualizar_optimas(o, reporte_falso("d", estados.CORRECTA, 1.2, "hd"), tol)
        self.assertEqual([x["id_candidata"] for x in o3], ["a", "b"])
        o4, c4, _ = actualizar_optimas(o, reporte_falso("e", estados.INCORRECTA, 0.001, "he"), tol)
        self.assertEqual((len(o4), c4), (2, 1.0), "una incorrecta nunca es óptima")
        o5, c5, mejora = actualizar_optimas(o, reporte_falso("f", estados.CORRECTA, 0.5, "hf"), tol)
        self.assertEqual(([x["id_candidata"] for x in o5], c5, mejora), (["f"], 0.5, True))


class TestMemoria(unittest.TestCase):
    """A05: historial de solo agregar, consultas y reconstrucción."""

    def test_operaciones(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "historial.jsonl"
            m = Memoria(ruta, 1e-9)
            m.agregar(reporte_falso("referencia", estados.CORRECTA, 2.0, "h0", True), 0)
            m.agregar(reporte_falso("it-0001", estados.FALLO_EVALUADOR, 10.0, "h1"), 1)
            m.agregar(reporte_falso("it-0002", estados.CORRECTA, 1.5, "h2"), 2)
            primera = ruta.read_text(encoding="utf-8").splitlines()[0]
            m.agregar(reporte_falso("it-0003", estados.CORRECTA, 1.7, "h2"), 3)
            m.agregar(reporte_falso("it-0004", estados.ERROR_INFRAESTRUCTURA, None, "h4"), 4)
            self.assertEqual(ruta.read_text(encoding="utf-8").splitlines()[0], primera)
            self.assertEqual(m.mejor_candidato()["id_evaluacion"], "it-0002")
            self.assertEqual(m.ya_evaluada("h2"), "it-0002")
            self.assertEqual(m.por_id("it-0003")["duplicada_de"], "it-0002")
            self.assertEqual(m.conteo()[estados.CORRECTA], 3)
            self.assertEqual(m.mejor_costo(), 1.5)
            with self.assertRaises(ValueError):
                m.agregar(reporte_falso("it-0002", estados.CORRECTA, 1.0, "h9"), 5)
            # Reanudar: otra Memoria sobre el mismo archivo reconstruye lo mismo.
            otra = Memoria(ruta, 1e-9).reconstruir()
            self.assertEqual([o["id_candidata"] for o in otra["optimas"]], ["it-0002"])
            # it-0003 no mejora (1); error_infraestructura no cuenta para la paciencia.
            self.assertEqual(otra["sin_mejora"], 1)

    def test_entrada_politica_es_pi_de_C_y_E(self):
        """π(C, E) = A: el agente recibe los costos de su última acción y el mejor candidato, no el historial."""
        with tempfile.TemporaryDirectory() as d:
            m = Memoria(Path(d) / "historial.jsonl", 1e-9)
            self.assertIsNone(m.entrada_politica())
            m.agregar(reporte_falso("referencia", estados.CORRECTA, 2.0, "h0", True), 0)
            m.agregar(reporte_falso("it-0001", estados.CORRECTA, 1.5, "h1"), 1)
            m.agregar(reporte_falso("it-0002", estados.CORRECTA, 1.8, "h2"), 2)
            entrada = m.entrada_politica()
            self.assertEqual(set(entrada), {"E", "mejor_candidato"})
            self.assertEqual(entrada["E"]["id"], "it-0002")
            self.assertEqual(set(entrada["E"]), {"id", "estado", "partio_de", "malla", "C_A", "C_L", "C_R", "C", "r"})
            # E son los cambios respecto al mejor candidato del que partió (it-0001, C = 1.5).
            self.assertEqual(entrada["E"]["partio_de"], "it-0001")
            self.assertEqual(entrada["E"]["C"]["antes"], 1.5)
            self.assertEqual(entrada["E"]["C"]["despues"], 1.8)
            self.assertAlmostEqual(entrada["E"]["C"]["cambio"], 0.3)
            # it-0002 dio un costo mayor: se descarta y el mejor candidato sigue siendo it-0001.
            self.assertEqual(entrada["mejor_candidato"]["id"], "it-0001")
            self.assertEqual(entrada["mejor_candidato"]["C"], 1.5)
            t = Temporal()
            try:
                prompt = construir_prompt(cargar_contexto(corrida(), t.instancia), entrada, "it-0003",
                                          Path(d) / "it-0003")
            finally:
                t.cerrar()
            self.assertIn("π(C, E) = A", prompt)
            self.assertIn("Mejor candidato actual", prompt)
            self.assertIn("cambios que produjo tu última acción", prompt)
            self.assertIn("se descarta", prompt)
            self.assertNotIn("H_t", prompt)
            self.assertNotIn("ultimos", prompt)


class TestHashes(unittest.TestCase):
    def test_hash_accion_es_el_cgra_json(self):
        """La acción se identifica por su cgra.json: candidata.json (justificación) no cuenta."""
        from rl.comun import hash_accion, hashes_paquete
        t = Temporal()
        try:
            a, b = t.config(("E", "V"), True, "a"), t.config(("E", "V"), True, "b")
            self.assertEqual(hash_accion(hashes_paquete(a)), hash_accion(hashes_paquete(b)))
            self.assertNotEqual(hash_accion(hashes_paquete(a)), hash_accion(hashes_paquete(t.config())))
        finally:
            t.cerrar()


if __name__ == "__main__":
    unittest.main()
