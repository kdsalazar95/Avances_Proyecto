"""Pruebas del evaluador E01–E08 con el simulador y el cargador reales."""

import json
import math
import unittest

from rl import estados
import shutil
from pathlib import Path

from rl.comun import FUENTE_OPERACIONES, leer_json, sha256_archivo
from rl.evaluador import funcional, metricas, recompensa, reporte, simulacion
from rl.evaluador.compilador import CompiladorReduccion, CompiladorSkills, crear_compilador, prompt_skills
from rl.evaluador.entradas import calcular_esperado, parametros_instancia
from rl.evaluador.evaluacion import evaluar
from rl.evaluador.validacion_programas import validar_programas

from .utilidades import CompiladorCaido, CompiladorMutado, Temporal, buscar, corrida, instruccion


class TestEntradas(unittest.TestCase):
    """La instancia sale de operaciones.c y el esperado se calcula en CPU."""

    @classmethod
    def setUpClass(cls):
        cls.t = Temporal()

    @classmethod
    def tearDownClass(cls):
        cls.t.cerrar()

    def test_memoria_generada_por_operaciones_c(self):
        """memoria.bin se genera compilando operaciones.c y tiene las regiones de la reducción."""
        self.assertEqual(parametros_instancia(FUENTE_OPERACIONES, "reduccion"), {"TAMANO_VECTOR": 100})
        esperado = calcular_esperado(self.t.instancia["memoria_inicial"], "reduccion")
        self.assertEqual(esperado["result"]["forma"], [1, 1])
        # a[i] = b[i] = i % 4 + 1  ->  c = 2·(1+2+3+4)·25 = 500
        self.assertEqual(esperado["result"]["valores"], [500.0])
        self.assertEqual(len(esperado["c"]["valores"]), 100)

    def test_esperado_matmul_y_convolucion(self):
        """Las referencias de matmul y convolución coinciden con un caso calculado a mano."""
        from rl.evaluador.entradas import esperado_convolucion, esperado_matmul
        r = esperado_matmul({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, {"a": (2, 2), "b": (2, 2)})
        self.assertEqual(r["result"], [19, 22, 43, 50])
        r = esperado_convolucion({"imagen": list(range(9)), "kernel": [1, 0, 0, 1]},
                                 {"imagen": (3, 3), "kernel": (2, 2)})
        self.assertEqual(r["result"], [0 + 4, 1 + 5, 3 + 7, 4 + 8])


class TestValidacionFuncional(unittest.TestCase):
    """E05 compara todos los elementos con eps_abs + eps_rel·|y|."""

    ESPERADO = {"r": {"forma": [1, 3], "valores": [1.0, 2.0, 4.0]}}

    def test_pasa(self):
        """Un resultado idéntico pasa y su C_R es 0."""
        r = funcional.comparar(self.ESPERADO, {"r": [1.0, 2.0, 4.0]}, {"r": (1, 3)}, 1e-6, 1e-6, 1e-9)
        self.assertTrue(r["aprobada"])
        self.assertEqual(r["C_R"], 0.0)

    def test_un_elemento_fuera(self):
        """Basta un elemento fuera de tolerancia para que sea incorrecta; C_R se conserva."""
        r = funcional.comparar(self.ESPERADO, {"r": [1.0, 2.0, 5.0]}, {"r": (1, 3)}, 1e-6, 1e-6, 1e-9)
        self.assertFalse(r["aprobada"])
        self.assertEqual(r["elementos_fallidos"], 1)
        self.assertAlmostEqual(r["C_R"], (1 / 4) / 3, places=6)

    def test_dimensiones_distintas_y_nan(self):
        """Una forma distinta o un NaN hacen la acción incorrecta."""
        r = funcional.comparar(self.ESPERADO, {"r": [1.0, 2.0]}, {"r": (1, 2)}, 1e-6, 1e-6, 1e-9)
        self.assertFalse(r["aprobada"])
        self.assertTrue(r["errores_estructurales"])
        r = funcional.comparar(self.ESPERADO, {"r": [1.0, math.nan, 4.0]}, {"r": (1, 3)}, 1e-6, 1e-6, 1e-9)
        self.assertFalse(r["aprobada"])
        self.assertIsNone(r["C_R"])


class TestMetricas(unittest.TestCase):
    """E06: casos calculados a mano."""

    PESOS = {"w1_MS": 4, "w2_MV": 8, "w3_CS": 1, "w4_CV": 2}

    def test_area_por_capacidades(self):
        """M = MS+MV = 12, C = CS+CV = 3, E = 1, V = 2, G = 15."""
        for instrucciones, esperado in ((["LD", "ST", "VLD", "VST"], 12), (["ADD", "VADD"], 3),
                                        (["ADD"], 1), (["VMUL"], 2), (["LD", "VST", "MUL", "VDIV"], 15)):
            conteo = dict.fromkeys(metricas.CAPACIDADES, 0)
            for cap in metricas.capacidades_tipo(instrucciones):
                conteo[cap] += 1
            self.assertEqual(metricas.costo_area(conteo, self.PESOS), esperado)

    def test_costo_total(self):
        """C = 1·45/81 + 2·184/245 + 3·0.01."""
        costo, a, l = metricas.costo_total(45, 184, 0.01, {"K_A": 1, "K_L": 2, "K_R": 3},
                                           {"C_A_ref": 81, "C_L_ref": 245})
        self.assertAlmostEqual(costo, 45 / 81 + 2 * 184 / 245 + 0.03)
        self.assertAlmostEqual(a, 45 / 81)

    def test_latencia_con_nops(self):
        """'ciclos_x_nops': ciclos·(1 + NOPs/(ciclos·PEs)); sin NOPs vale los ciclos y más NOPs cuestan más."""
        self.assertEqual(metricas.costo_latencia(100, 0, "ciclos_x_nops", 4), 100)
        self.assertAlmostEqual(metricas.costo_latencia(100, 200, "ciclos_x_nops", 4), 150)   # mitad en NOP
        self.assertAlmostEqual(metricas.costo_latencia(641, 874, "ciclos_x_nops", 4), 641 + 874 / 4)
        self.assertEqual(metricas.costo_latencia(100, 200, "ciclos", 4), 100)
        self.assertAlmostEqual(metricas.fraccion_nops(100, 200, 4), 0.5)
        with self.assertRaises(metricas.ErrorMetricas):
            metricas.costo_latencia(100, None, "ciclos_x_nops", 4)

    def test_sin_parametros_es_error(self):
        """Sin K o sin normalizadores no se calcula con valores ocultos."""
        with self.assertRaises(metricas.ErrorMetricas):
            metricas.costo_total(1, 1, 0, {"K_A": 1, "K_L": 1}, {"C_A_ref": 1, "C_L_ref": 1})
        with self.assertRaises(metricas.ErrorMetricas):
            metricas.costo_total(1, 1, 0, {"K_A": 1, "K_L": 1, "K_R": 1}, None)


class TestRecompensa(unittest.TestCase):
    """E07: r = C_best_prev − C, −P_fallo o None."""

    def test_casos(self):
        """Primera sin referencia, mejora, empeora, empate, penalizados y fallo de infraestructura."""
        with self.assertRaises(ValueError):
            recompensa.recompensa(estados.CORRECTA, 1.0, None, 10)
        self.assertAlmostEqual(recompensa.recompensa(estados.CORRECTA, 0.9, 1.0, 10)[0], 0.1)
        self.assertAlmostEqual(recompensa.recompensa(estados.CORRECTA, 1.2, 1.0, 10)[0], -0.2)
        self.assertEqual(recompensa.recompensa(estados.CORRECTA, 1.0, 1.0, 10)[0], 0.0)
        for estado in estados.PENALIZADOS:
            self.assertEqual(recompensa.recompensa(estado, None, 1.0, 10)[0], -10)
        self.assertIsNone(recompensa.recompensa(estados.ERROR_INFRAESTRUCTURA, None, 1.0, 10)[0])
        self.assertIn(estados.FALLO_EVALUADOR, estados.PENALIZADOS)
        self.assertTrue(recompensa.verificar_penalizacion(1.5, [1.0, 2.0]))
        self.assertFalse(recompensa.verificar_penalizacion(10, [1.0, 2.0]))


class TestReporteCiclos(unittest.TestCase):
    def test_lee_ciclos_y_nops(self):
        """Ciclos y NOP salen del reporte_ciclos.txt del simulador."""
        texto = ("Total de ciclos:        24\n"
                 "  PE    Tipo      Memoria  Cálculo  Comunic.  Otras    NOP  Ocupado\n"
                 "  PE00  generico       13        8         2      1      0   100.0%\n"
                 "  PE01  generico       12        6         1      1      4    83.3%\n")
        self.assertEqual(simulacion.leer_reporte(texto), (24, 4, {"PE00": 0, "PE01": 4}))
        self.assertEqual(simulacion.leer_reporte("nada"), (None, None, None))


class TestValidarProgramas(unittest.TestCase):
    """E03: "¿Están bien los .txt?" sobre los PE*.txt que generaron los skills."""

    @classmethod
    def setUpClass(cls):
        cls.t = Temporal()
        # 'M E M' x 3: programas escalares (los de la referencia E/V se prueban en TestEvaluador).
        cls.cgra = cls.t.config(("E", "E", "E"), True, "escalar") / "cgra.json"
        cls.memoria = cls.t.instancia["memoria_inicial"]

    @classmethod
    def tearDownClass(cls):
        cls.t.cerrar()

    def _validar(self, programas):
        return validar_programas(self.cgra, programas / "pe_instructions", self.memoria,
                                 leer_json(programas / "manifest.json"), ["c", "result"])

    def test_programas_correctos(self):
        r = self._validar(self.t.programas(self.cgra))
        self.assertTrue(r["valida"], r["errores"])
        self.assertEqual(r["ciclos_programa"], 247)

    def test_instruccion_fuera_del_tipo(self):
        """ADD en un PE M lo rechaza el cargador real, con el PE identificado."""
        p = self.t.programas(self.cgra)
        instruccion(p / "pe_instructions" / "PE02.txt", 0, "ADD rA, rA, rA")
        r = self._validar(p)
        self.assertFalse(r["valida"])
        self.assertTrue(any("PE02" in (e.get("archivo") or "") for e in r["errores"]))

    def test_transferencia_sin_pareja(self):
        p = self.t.programas(self.cgra)
        ruta = p / "pe_instructions" / "PE11.txt"
        ciclo = buscar(ruta, "RECV west, rA")
        instruccion(ruta, ciclo, "NOP")
        r = self._validar(p)
        self.assertFalse(r["valida"])
        self.assertTrue(any(e.get("ciclo") == ciclo for e in r["errores"]))

    def test_registros_y_cobertura(self):
        """Registro leído antes de escribirse, salida sin escribir, salida duplicada y fuera de rango."""
        p = self.t.programas(self.cgra)
        instruccion(p / "pe_instructions" / "PE11.txt", 0, "NOP")
        ruta = p / "pe_instructions" / "PE00.txt"
        instruccion(ruta, buscar(ruta, "ST c[3]"), "ST c[0], rC")
        reglas = {e["regla"] for e in self._validar(p)["errores"]}
        self.assertTrue({"registro_sin_definir", "salida_sin_cubrir", "salida_duplicada"} <= reglas, reglas)
        p = self.t.programas(self.cgra)
        ruta = p / "pe_instructions" / "PE00.txt"
        instruccion(ruta, buscar(ruta, "LD rA, a[0]"), "LD rA, a[100]")
        self.assertTrue(any("se sale de la región" in e["detalle"] for e in self._validar(p)["errores"]))

    def test_sobrante_encabezado_y_manifest(self):
        p = self.t.programas(self.cgra)
        (p / "pe_instructions" / "PE33.txt").write_text("00: NOP\n", encoding="utf-8")
        texto = (p / "pe_instructions" / "PE01.txt").read_text(encoding="utf-8")
        (p / "pe_instructions" / "PE01.txt").write_text(texto.replace("(E)", "(M)"), encoding="utf-8")
        m = leer_json(p / "manifest.json")
        m["filas"] = 2
        (p / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        reglas = {e["regla"] for e in self._validar(p)["errores"]}
        self.assertTrue({"pe_sobrante", "encabezado_incoherente", "manifest_incoherente"} <= reglas, reglas)


class TestEvaluador(unittest.TestCase):
    """Evaluador completo: Skills 01–09 (sustituto) + simulador real + medición."""

    @classmethod
    def setUpClass(cls):
        cls.t = Temporal()
        cls.corrida = corrida()
        cls.ref_dir = cls.t.config(("E", "V", "E"), True, "referencia")
        cls.ref = evaluar(cls.ref_dir, "referencia", cls.t.instancia, cls.corrida, None, None,
                          cls.t.carpeta("eval-ref"), CompiladorReduccion())
        cls.norm = cls.ref["parametros"]["normalizadores"]

    @classmethod
    def tearDownClass(cls):
        cls.t.cerrar()

    def _evaluar(self, accion, compilador=None, nombre="x", corrida_=None):
        return evaluar(accion, nombre, self.t.instancia, corrida_ or self.corrida, self.norm, 2.0,
                       self.t.carpeta(f"eval-{nombre}"), compilador or CompiladorReduccion())

    def test_referencia_calculada_a_mano(self):
        """Referencia 'M E M / M V M / M E M' (sin C).

        C_A = 6 M·12 + 2 E·1 + 1 V·2 = 76.
        Ciclos = 2 (inicio) + 7·17 (la fila V hace 17 bloques de 4; cada E, 4 bloques en 4·7·4 = 112)
                 + 1 (VREDSUM) + 2 (V -> E0 y suma) + 3 (E2 -> E0 en dos saltos y suma) + 2 (guardar) = 129.
        """
        self.assertEqual(self.ref["estado_evaluacion"], estados.CORRECTA)
        self.assertEqual((self.ref["metricas"]["C_A"], self.ref["metricas"]["ciclos"]), (76, 129))
        self.assertEqual(self.ref["metricas"]["capacidades"], {"MS": 6, "MV": 6, "CS": 2, "CV": 1})
        self.assertEqual(self.ref["costo_total"], 2.0)
        self.assertEqual(self.norm["C_A_ref"], 76)
        manifest = leer_json(Path(self.ref["programas_generados"]["directorio"]).parent / "manifest.json")
        bloques = [len(a["bloques_vectoriales"]) for a in manifest["asignacion"].values()]
        self.assertEqual(bloques, [0, 17, 0])
        self.assertEqual(self.ref["programas_generados"]["archivos"], 9)
        self.assertFalse(reporte.comprobar(self.ref))

    def test_accion_es_solo_el_cgra_json(self):
        """El evaluador recibe solo el cgra.json: lo copia sin cambios y los skills generan los PE*.txt."""
        accion = self.t.config(("V", "V", "V"), True)
        antes = sha256_archivo(accion / "cgra.json")
        r = self._evaluar(accion, nombre="mc")
        self.assertEqual(r["accion"]["hashes"], {"cgra.json": antes})
        self.assertEqual(r["accion"]["hash_accion"], antes)
        self.assertEqual(sha256_archivo(accion / "cgra.json"), antes)
        self.assertFalse((accion / "pe_instructions").exists())
        self.assertEqual(r["estado_evaluacion"], estados.CORRECTA)
        # 3x3 'M V M': C_A = 6·12 + 3·2 = 78; 73 ciclos -> C = 78/76 + 73/129
        self.assertEqual((r["metricas"]["C_A"], r["metricas"]["ciclos"]), (78, 73))
        self.assertAlmostEqual(r["costo_total"], 78 / 76 + 73 / 129)
        self.assertAlmostEqual(r["recompensa"], 2.0 - (78 / 76 + 73 / 129))

    def test_skills_reportan(self):
        """Si los skills no saben mapear la configuración ("Reportar") es fallo_evaluador penalizado."""
        accion = self.t.config()
        texto = (accion / "cgra.json").read_text(encoding="utf-8").replace('"M E M"', '"M M E"', 1)
        (accion / "cgra.json").write_text(texto, encoding="utf-8")
        r = self._evaluar(accion, nombre="reporta")
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_EVALUADOR)
        self.assertEqual(r["errores"][0]["regla"], "skills_reportan")
        self.assertEqual((r["costo_total"], r["recompensa"]), (10.0, -10.0))
        self.assertIsNone(r["metricas"]["ciclos"])

    def test_reintento_con_errores(self):
        """Si los .txt fallan la validación, los skills reciben los errores y reintentan (N_corr)."""
        def mutar(destino, intento):
            if intento == 1:
                instruccion(destino / "pe_instructions" / "PE02.txt", 0, "ADD rA, rA, rA")
        compilador = CompiladorMutado(mutar)
        r = self._evaluar(self.t.config(("E", "E", "E")), compilador, "reintento")
        self.assertEqual(r["estado_evaluacion"], estados.CORRECTA)
        self.assertEqual(r["programas_generados"]["intentos"], 2)
        self.assertIsNone(compilador.intentos[0])
        self.assertTrue(any("PE02" in e for e in compilador.intentos[1]))

    def test_txt_siguen_mal_tras_n_corr(self):
        """Si siguen mal después de N_corr correcciones: Reportar -> fallo_evaluador."""
        mutar = lambda d, i: instruccion(d / "pe_instructions" / "PE02.txt", 0, "ADD rA, rA, rA")
        compilador = CompiladorMutado(mutar)
        r = self._evaluar(self.t.config(("E", "E", "E")), compilador, "nocorr")
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_EVALUADOR)
        self.assertEqual(len(compilador.intentos), 2)  # N_corr = 1 -> 1 + 1 intentos
        self.assertEqual(r["errores"][-1]["regla"], "reportar")

    def test_resultado_incorrecto(self):
        """Programas que terminan con un resultado equivocado: incorrecta, costo penalizado, C_R conservado."""
        mutar = lambda d, i: instruccion(d / "pe_instructions" / "PE11.txt", 0, "MOV acc, 1.0")
        r = self._evaluar(self.t.config(("E", "E", "E")), CompiladorMutado(mutar), "incorrecta")
        self.assertEqual(r["estado_evaluacion"], estados.INCORRECTA)
        self.assertEqual(r["validacion_funcional"]["elementos_fallidos"], 1)
        self.assertEqual(r["metricas"]["ciclos"], 247)
        self.assertEqual((r["costo_total"], r["recompensa"]), (10.0, -10.0))
        self.assertGreater(r["metricas"]["C_R"], 0)

    def test_fallo_del_simulador(self):
        """Una división entre cero aborta el simulador: fallo_simulador sin métricas inventadas."""
        def mutar(d, i):
            ruta = d / "pe_instructions" / "PE01.txt"
            instruccion(ruta, buscar(ruta, "ADD rC, rA, rB"), "DIV rC, rA, rT")
            instruccion(ruta, buscar(ruta, "NOP"), "MOV rT, 0.0")
        r = self._evaluar(self.t.config(("E", "E", "E")), CompiladorMutado(mutar), "div0")
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_SIMULADOR, r["errores"])
        self.assertIsNone(r["metricas"]["ciclos"])
        self.assertEqual(r["recompensa"], -10.0)

    def test_tiempo_maximo(self):
        corta = corrida(limites={**self.corrida["limites"], "t_max_simulacion_s": 0.001})
        r = self._evaluar(self.t.config(("V", "V", "V"), True), nombre="lento", corrida_=corta)
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_SIMULADOR)
        self.assertEqual(r["errores"][0]["regla"], "tiempo_excedido")

    def test_error_de_infraestructura(self):
        """Si el LLM de los skills no está disponible no se castiga al agente."""
        r = self._evaluar(self.t.config(), CompiladorCaido(), "caido")
        self.assertEqual(r["estado_evaluacion"], estados.ERROR_INFRAESTRUCTURA)
        self.assertIsNone(r["recompensa"])
        self.assertIsNone(r["costo_total"])

    def test_compilador_por_comando(self):
        """CompiladorSkills llama a un LLM externo (aquí un script) con el prompt de los skills."""
        script = self.t.raiz / "skills_falsos.py"
        script.write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.getcwd())\n"
            "from pathlib import Path\n"
            "from rl.evaluador.compilador_sustituto import compilar\n"
            "prompt, destino, cgra = Path(sys.argv[1]).read_text(), sys.argv[2], sys.argv[3]\n"
            "assert 'skills 01 a 09' in prompt and 'No modifiques' in prompt\n"
            "compilar(cgra, 'reduccion', 100, destino)\n"
            "Path(destino, 'reporte_skills.txt').write_text('Compilación completada')\n", encoding="utf-8")
        import sys
        compilador = CompiladorSkills([sys.executable, str(script), "{prompt_archivo}", "{dir_salida}", "{cgra}"])
        # Con PE*.txt, un reporte_skills.txt (aunque sea un resumen de éxito) no es un fallo: lo hizo Codex.
        r = self._evaluar(self.t.config(), compilador, "comando")
        self.assertEqual(r["estado_evaluacion"], estados.CORRECTA, r["errores"])
        self.assertTrue(list((self.t.raiz).rglob("intento-1.log")), "el log de los skills se escribe en vivo")
        solo_reporte = self.t.raiz / "solo_reporte.py"
        solo_reporte.write_text("import sys\nfrom pathlib import Path\nPath(sys.argv[1]).mkdir(parents=True, exist_ok=True)\n"
                                "Path(sys.argv[1], 'reporte_skills.txt').write_text('no se puede mapear')\n",
                                encoding="utf-8")
        r = self._evaluar(self.t.config(), CompiladorSkills([sys.executable, str(solo_reporte), "{dir_salida}"]),
                          "solo-reporte")
        self.assertEqual(r["estado_evaluacion"], estados.FALLO_EVALUADOR)
        caido = CompiladorSkills(["comando-que-no-existe-xyz"])
        r = self._evaluar(self.t.config(), caido, "sin-llm")
        self.assertEqual(r["estado_evaluacion"], estados.ERROR_INFRAESTRUCTURA)
        with self.assertRaises(ValueError):
            crear_compilador({"tipo": "otro"})

    def test_prompt_de_los_skills(self):
        texto = prompt_skills(self.ref_dir / "cgra.json", self.t.instancia, self.t.raiz / "x", ["[regla] PE01 mal"])
        self.assertIn("**solo** la operación `reduccion`", texto)
        self.assertIn("PE01 mal", texto)

    def test_reporte_guardado(self):
        guardado = leer_json(next(self.t.raiz.glob("eval-ref-*")) / "reporte.json")
        self.assertEqual(guardado["estado_evaluacion"], estados.CORRECTA)
        self.assertEqual(guardado["parametros"]["K"], self.corrida["K"])
        json.dumps(guardado)


if __name__ == "__main__":
    unittest.main()
