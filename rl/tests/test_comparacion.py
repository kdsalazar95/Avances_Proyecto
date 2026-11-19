"""Comparación entre sistemas: normalizadores fijos, epsilon (paso al azar), episodios, caché, convergencia,
corte por paciencia y barrido. Con el sustituto de los skills (sin LLM)."""

import csv
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl import estados
from rl.__main__ import main as cli
from rl.agente import ciclo as modulo_ciclo
from rl.agente.ciclo import ErrorCiclo, iniciar
from rl.agente.episodios import movimientos, paso_al_azar
from rl.agente.politica import Politica, PoliticaEpsilon, PoliticaGuion, _propuesta
from rl.comun import ESQUEMA_CGRA, escribir_json, leer_json, leer_jsonl
from rl.corrida import ErrorCorrida
from rl.entorno.validacion_accion import validar_accion
from rl.herramientas import convergencia
from rl.herramientas.barrido import barrido, leer_epsilons, nombre
from rl.herramientas.convergencia import COLUMNAS, escribir
from rl.herramientas.referencia import escribir_cgra, malla_filas

from .utilidades import corrida

POR_EPISODIOS = {"max_episodios": 6, "paciencia": 2, "sin_mejora_episodio": 2, "max_pasos_episodio": 10,
                 "t_max_simulacion_s": 60}


class LLMFalso(Politica):
    """Hace de LLM: propone, en ciclo, las configuraciones de ``carpetas``; guarda lo que recibe (``entradas``).
    ``parar_en``: ids en los que responde PARAR."""

    def __init__(self, carpetas, parar_en=()):
        self.carpetas, self.k, self.entradas, self.parar_en = list(carpetas), 0, [], set(parar_en)

    def proponer(self, contexto, entrada, id_candidata, dir_salida):
        self.entradas.append((id_candidata, entrada))
        if id_candidata in self.parar_en:
            Path(dir_salida).mkdir(parents=True)
            escribir_json(Path(dir_salida) / "candidata.json", {"id_candidata": id_candidata, "parar": True})
            return _propuesta(id_candidata, dir_salida)
        shutil.copytree(self.carpetas[self.k % len(self.carpetas)], dir_salida)
        self.k += 1
        return _propuesta(id_candidata, dir_salida)


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
        """La corrida de prueba; sin tabla, curva ni GIF al terminar (``informes``), salvo que se pidan."""
        ruta = self.raiz / nombre_
        escribir_json(ruta, corrida(**{"informes": False, **cambios}))
        return ruta

    def guion(self, *nombres):
        return PoliticaGuion([self.acciones / n for n in nombres])

    def llm(self, *nombres, **k):
        return LLMFalso([self.acciones / n for n in nombres], **k)


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
        """rl/config/comparacion.json: la 32x32 de cgra.json (fila 0 E), normalizadores fijos, por episodios con la
        paciencia y el tope del Sistema 2 (3 y 10) y 36000 s de simulación."""
        from rl.corrida import cargar_corrida, normalizadores_fijos
        c = cargar_corrida(Path(__file__).resolve().parents[1] / "config" / "comparacion.json")
        malla = c["referencia"]["generar"]["malla"]
        self.assertEqual((len(malla), len(malla[0].split())), (32, 32))
        self.assertEqual(malla[0].split()[:3] + malla[1].split()[:2], ["M", "E", "E", "M", "V"])
        self.assertTrue(all(f.split()[0] == f.split()[-1] == "M" for f in malla))
        lim = c["limites"]
        self.assertEqual((lim["max_episodios"], lim["paciencia"], lim["sin_mejora_episodio"],
                          lim["t_max_simulacion_s"]), (10, 3, 3, 36000))
        self.assertNotIn("requiere_escalar_y_vectorial", c["restricciones"])
        self.assertEqual(normalizadores_fijos(c)["operaciones"]["matmul"]["C_L_ref"], 904.8888888888889)
        self.assertEqual(c["politica"]["tipo"], "comando")


class TestEpsilon(_Base):

    def test_1_epsilon_0_es_el_llm_y_1_es_un_paso_al_azar(self):
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
        for e in h:  # cada paso al azar es un movimiento desde el mejor candidato y cumple las restricciones
            self.assertNotEqual(e["estado_evaluacion"], estados.ACCION_INVALIDA, e["errores"][:1])
            self.assertIn("exploración al azar", e["justificacion_agente"])

    def test_2_mezcla_y_misma_semilla_mismas_decisiones(self):
        def decisiones(semilla, carpeta):
            c = iniciar(self.config(id=carpeta), self.raiz / "corridas")
            c.correr(PoliticaEpsilon(self.guion("V3", "V4", "V2", "E3"), 0.5, semilla, ESQUEMA_CGRA),
                     max_iteraciones=4)
            return [e["decidio"] for e in c.memoria.entradas[1:]], [e["malla"] for e in c.memoria.entradas[1:]]

        a = decisiones(3, "a")
        self.assertEqual(a, decisiones(3, "b"))
        self.assertEqual(set(a[0]), {"llm", "exploracion"})

    def test_3_movimientos_del_sistema_2(self):
        """Desde la 32x32: quitar 16, 8, 4, 2 o 1 filas o columnas del interior (inicio o final) o cambiar un PE
        del interior E <-> V. Todos cumplen las restricciones; el azar elige primero la clase y luego el movimiento."""
        ciclo = iniciar(self.config(restricciones={"memoria_solo_en_bordes_oeste_este": True,
                                                   "requiere_memoria_y_computo": True,
                                                   "filas": {"minimo": 1, "maximo": 32},
                                                   "columnas": {"minimo": 3, "maximo": 32}}),
                        self.raiz / "corridas")
        r = ciclo.contexto["restricciones"]
        malla = [("M " + " ".join(("E" if f % 2 == 0 else "V") for _ in range(30)) + " M") for f in range(32)]
        m = movimientos(malla, r)
        # filas alternas E/V: quitar un número par de filas del inicio o del final da la misma malla; todas las
        # columnas del interior son iguales: del inicio o del final da lo mismo. Cuentan una vez (el del final).
        self.assertEqual([x[0] for x in m["bloque"]],
                         ["FILAS FIN 16", "COLUMNAS FIN 16", "FILAS FIN 8", "COLUMNAS FIN 8", "FILAS FIN 4",
                          "COLUMNAS FIN 4", "FILAS FIN 2", "COLUMNAS FIN 2", "FILAS FIN 1", "FILAS INICIO 1",
                          "COLUMNAS FIN 1"])
        self.assertEqual(len(m["tipo"]), 32 * 30)
        self.assertIn(("FILAS FIN 16", malla[:16]), m["bloque"])
        nombre_, cols = next(x for x in m["bloque"] if x[0] == "COLUMNAS FIN 8")
        self.assertEqual(len(cols[0].split()), 24)
        for nombre_, nueva in m["bloque"] + m["tipo"][:40]:
            destino = escribir_cgra(self.raiz / "mov" / nombre_.replace(" ", "_").replace(">", ""), nueva).parent
            self.assertTrue(validar_accion(destino, ciclo.contexto, "x")["paquete_valido"], nombre_)
        azar = random.Random(0)
        clases = [paso_al_azar(azar, malla, r)[0].split()[0] for _ in range(200)]
        self.assertGreater(clases.count("CAMBIAR"), 60)  # ~50 % cada clase, no ~98 % cambios de PE
        self.assertGreater(len(clases) - clases.count("CAMBIAR"), 60)
        # desde una malla de una fila y 3 columnas solo se puede cambiar el PE
        self.assertEqual(movimientos(["M E M"], r)["bloque"], [])


class TestEpisodios(_Base):

    def test_1_episodios_con_la_regla_del_sistema_2(self):
        """Cada episodio empieza en la referencia y termina tras sin_mejora_episodio propuestas sin mejorar su mejor;
        la corrida para con paciencia episodios seguidos sin bajar el mejor C."""
        ciclo = iniciar(self.config(epsilon=1.0, semilla=5, limites=POR_EPISODIOS), self.raiz / "corridas")
        estado = ciclo.correr(PoliticaEpsilon(self.guion(), 1.0, 5, ESQUEMA_CGRA))
        self.assertIn(estado, ("terminada: paciencia", "terminada: max_episodios"))
        datos = leer_json(ciclo.carpeta / "episodios.json")
        eps = datos["episodios"]
        self.assertEqual([ep["n"] for ep in eps], list(range(len(eps))))
        h = ciclo.memoria.entradas
        ref = h[0]
        for ep in eps:
            propias = [e for e in h if (e.get("episodio") or {}).get("n") == ep["n"]]
            self.assertEqual(len(propias), ep["pasos"])
            self.assertEqual([e["episodio"]["paso"] for e in propias], list(range(ep["pasos"])))
            self.assertIn(ep["fin"], ("sin_mejora", "tope"))
            # el primer paso sale de la referencia (3 filas 'M X M'): un movimiento desde ella
            primera = propias[0]["malla"]
            self.assertTrue(len(primera) < 3 or sum(a != b for a, b in zip(primera, ref["malla"])) == 1, primera)
        # paciencia: episodios seguidos sin mejora al final
        if estado == "terminada: paciencia":
            self.assertTrue(all(not ep["mejora"] for ep in eps[-2:]))
        self.assertEqual(datos["C_best"], min(ep["C"] for ep in eps))
        self.assertEqual(datos["motivo_fin"], estado)

    def test_2_el_llm_ve_el_episodio_y_los_anteriores_y_puede_parar(self):
        llm = self.llm("V3", "V2", parar_en={"it-0003"})
        ciclo = iniciar(self.config(limites=dict(POR_EPISODIOS, max_episodios=2)), self.raiz / "corridas")
        ciclo.correr(PoliticaEpsilon(llm, 0.0, 0, ESQUEMA_CGRA))
        eps = leer_json(ciclo.carpeta / "episodios.json")["episodios"]
        self.assertEqual(eps[0]["fin"], "parar")
        self.assertEqual(eps[0]["pasos"], 2)
        primero = llm.entradas[0][1]
        self.assertIsNone(primero["E"])  # el episodio empieza en la referencia
        self.assertEqual(primero["episodio"]["mejor"]["id"], "referencia")
        self.assertEqual(primero["episodio"]["anteriores"], [])
        segundo_episodio = next(e for i, e in llm.entradas if e["episodio"]["n"] == 1)
        self.assertEqual(segundo_episodio["episodio"]["anteriores"][0]["C"], eps[0]["C"])
        self.assertIsNone(segundo_episodio["E"])
        self.assertTrue((ciclo.carpeta / "candidatas" / "it-0003.parar-0").is_dir())
        # el prompt explica el episodio y cómo parar
        from rl.agente.politica import construir_prompt
        texto = construir_prompt(ciclo.contexto, segundo_episodio, "it-0009", ciclo.carpeta / "x")
        self.assertIn("### Episodio", texto)
        self.assertIn('"parar": true', texto)

    def test_3_cache_una_configuracion_repetida_no_se_vuelve_a_evaluar(self):
        """La misma configuración (aunque cambie la descripción del cgra.json) no se vuelve a evaluar: el reporte
        sale de la evaluación anterior, con su recompensa recalculada, y en la convergencia cuesta 0 s."""
        ciclo = iniciar(self.config(limites=dict(POR_EPISODIOS, max_episodios=2)), self.raiz / "corridas")
        with mock.patch.object(modulo_ciclo, "evaluar", wraps=modulo_ciclo.evaluar) as evaluar:
            ciclo.correr(PoliticaEpsilon(self.llm("V3", "V2", "V3"), 0.0, 0, ESQUEMA_CGRA))
        h = ciclo.memoria.entradas[1:]
        claves = [e["clave_hardware"] for e in h]
        self.assertEqual(evaluar.call_count, len(set(claves)))  # una evaluación por configuración distinta
        repetidas = [e for e in h if e.get("repetida")]
        self.assertTrue(repetidas)
        self.assertTrue(all(e["reusada_de"].startswith("it-") for e in repetidas))
        filas = convergencia.filas_corrida(ciclo.carpeta)
        for f in filas:
            e = ciclo.memoria.por_id(f["id"])
            if e.get("repetida"):
                self.assertEqual(f["tiempo_evaluacion_s"], 0.0)


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

    def test_2_el_corte_por_paciencia_da_lo_mismo_que_correr_con_esa_paciencia(self):
        def correr(id_, paciencia):
            c = iniciar(self.config(f"{id_}.json", id=id_, epsilon=1.0, semilla=2,
                                    limites=dict(POR_EPISODIOS, max_episodios=30, paciencia=paciencia)),
                        self.raiz / "corridas")
            c.correr(PoliticaEpsilon(self.guion(), 1.0, 2, ESQUEMA_CGRA))
            return c.carpeta

        larga, corta = correr("larga", 4), correr("corta", 1)
        eps = leer_json(larga / "episodios.json")
        n, motivo = convergencia.corte(eps["episodios"], 1, eps["motivo_fin"])
        real = leer_json(corta / "episodios.json")
        self.assertEqual((n, motivo), (len(real["episodios"]), "paciencia"))

        def sin_tiempos(filas):
            return [{k: v for k, v in f.items() if not k.startswith("tiempo") and k != "corrida"} for f in filas]
        self.assertEqual(sin_tiempos(convergencia.filas_corrida(larga, n)), sin_tiempos(convergencia.filas_corrida(corta)))
        por_p = leer_json(larga / "convergencia.json")["por_paciencia"]
        self.assertEqual(sorted(por_p, key=int), ["3", "4"])  # las <= 4 de PACIENCIAS, y la de la corrida
        self.assertLessEqual(por_p["3"]["episodios"], por_p["4"]["episodios"])


class TestResultados(_Base):

    def test_1_tabla_curva_gif_y_graficas(self):
        """Al terminar una corrida por episodios: configuraciones.csv (las columnas del Sistema 2, una fila por
        configuración distinta, el frente de Pareto marcado), curva_costos.png y exploracion.gif; y las gráficas de
        la comparación a partir de comparacion_epsilon.csv y curvas_epsilon.csv."""
        from rl.herramientas import tabla
        ciclo = iniciar(self.config(epsilon=0.5, semilla=1, informes=True, limites=dict(POR_EPISODIOS, max_episodios=3)),
                        self.raiz / "corridas")
        ciclo.correr(PoliticaEpsilon(self.llm("V3", "V2", "E3"), 0.5, 1, ESQUEMA_CGRA))
        with (ciclo.carpeta / "configuraciones.csv").open(encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        self.assertEqual(list(filas[0])[:len(tabla.BASE)], ["configuracion"] + tabla.BASE[1:])
        claves = {e["clave_hardware"] for e in ciclo.memoria.entradas if e.get("clave_hardware")}
        self.assertEqual(len(filas), len(claves))
        self.assertTrue(any(f["eficiente"] == "True" for f in filas))
        self.assertEqual(sum(f["origen"] == "referencia" for f in filas), 1)
        self.assertIn("[E/V/E]", next(f["configuracion"] for f in filas if f["origen"] == "referencia"))
        for archivo in ("curva_costos.png", "exploracion.gif", "configuraciones.json", "configuraciones_operaciones.csv"):
            self.assertTrue((ciclo.carpeta / archivo).is_file(), archivo)
        from PIL import Image
        with Image.open(ciclo.carpeta / "exploracion.gif") as gif:
            self.assertEqual(gif.n_frames, len(ciclo.memoria.entradas) + 1)  # una por evaluación y la mejor
        # gráficas de la comparación
        salida = self.raiz / "comp"
        convergencia.comparar([ciclo.carpeta], salida, paciencias=[1, 2])
        self.assertEqual(cli(["graficas", str(salida), "--salida", str(salida)]), 0)
        for archivo in ("convergencia_por_evaluacion.png", "convergencia_por_tiempo.png", "tiempo_convergencia.png",
                        "mejor_costo.png"):
            self.assertTrue((salida / archivo).is_file(), archivo)


class TestBarrido(_Base):

    def test_1_una_corrida_por_epsilon_con_la_misma_referencia_y_cache(self):
        self.assertEqual(leer_epsilons(["0", "10", "20", "80", "90", "100"]), [0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        self.assertEqual(nombre("s1", 0.1), "s1-eps010")
        config = self.config(id="s1", limites=POR_EPISODIOS)
        corridas = self.raiz / "corridas"

        def politica(c):
            return PoliticaEpsilon(self.llm("V2", "V3"), c.corrida["epsilon"], 0, ESQUEMA_CGRA)

        carpetas, resumenes = barrido(config, [0.0, 1.0], corridas, politica=politica, paciencia=3, max_episodios=6,
                                      paciencias=[1, 2, 3])
        self.assertEqual([c.name for c in carpetas], ["s1-eps000", "s1-eps100"])
        ref0, ref1 = (leer_jsonl(c / "historial.jsonl")[0] for c in carpetas)
        self.assertNotIn("reusada_de", ref0)
        self.assertIn("s1-eps000", ref1["reusada_de"])  # la referencia se evaluó una sola vez
        self.assertEqual(ref0["costo_total"], ref1["costo_total"])
        corrida1 = leer_json(carpetas[1] / "corrida.json")
        self.assertEqual((corrida1["limites"]["paciencia"], corrida1["limites"]["max_episodios"]), (3, 6))
        self.assertTrue(any((corridas / "s1-cache").iterdir()))  # caché compartida entre corridas
        self.assertEqual([(r["epsilon"], r["paciencia"]) for r in resumenes],
                         [(0.0, 1), (0.0, 2), (0.0, 3), (1.0, 1), (1.0, 2), (1.0, 3)])
        salida = corridas / "s1-comparacion"
        with (salida / "comparacion_epsilon.csv").open(encoding="utf-8") as f:
            tabla = list(csv.DictReader(f))
        self.assertEqual(tuple(tabla[0]), convergencia.COLUMNAS_COMPARACION)
        self.assertEqual(len(tabla), 6)
        self.assertEqual(tabla[-1]["fraccion_exploracion"], "1.0")
        # reanudar: las corridas terminadas no proponen nada más
        barrido(config, [0.0, 1.0], corridas, politica=lambda c: PoliticaGuion([]), paciencia=3, max_episodios=6,
                paciencias=[1, 2, 3])
        # la CLI junta corridas
        self.assertEqual(cli(["comparar", str(carpetas[0]), str(carpetas[1]), "--salida", str(self.raiz / "c")]), 0)
        self.assertTrue((self.raiz / "c" / "comparacion_epsilon.csv").is_file())

    def test_2_politica_manual_no_sirve_para_el_barrido(self):
        with self.assertRaises(ErrorCiclo):
            barrido(self.config(), [0.0, 0.5], self.raiz / "corridas")
        self.assertFalse((self.raiz / "corridas").exists())


if __name__ == "__main__":
    unittest.main()
