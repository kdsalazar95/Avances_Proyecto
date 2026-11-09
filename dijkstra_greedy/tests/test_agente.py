"""Pruebas del agente Greedy (dijkstra_greedy/dijkstra/agente.py): cadena de mejoras
determinística de configuraciones de la CGRA, con un evaluador falso y con el real."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_raiz_dg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(_raiz_dg), "cgra", "src"))  # simulador (cgra/src)
sys.path.insert(0, _raiz_dg)  # Dijkstra + Greedy (dijkstra_greedy/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dijkstra.agente import Agente, EvaluadorParalelo, Parametros, Registro
from dijkstra.compilar import FUENTE, ErrorCompilacion, preparar
from dijkstra.evaluador import RUTA_COSTOS, Evaluacion, Evaluador
from formato_cgra import RUTA_POR_DEFECTO
from run_cgra import SimuladorCGRA
from test_dijkstra import _ConOperaciones

with open(RUTA_POR_DEFECTO, encoding="utf-8") as _archivo:
    BASE = json.load(_archivo)

AREA = {"G": 15, "M": 12, "E": 1, "V": 2, "C": 3}
POTENCIA = {"G": 3, "E": 1, "V": 2, "C": 3}
MALLA_3X3 = (("M", "E", "M"), ("M", "V", "M"), ("M", "C", "M"))


def medir_falso(k_area=1.0, k_latencia=1.0, invalida=None, llamadas=None):
    """Evaluador falso y rápido: área con los pesos de costos.json y una latencia que
    baja con la potencia de cómputo y con la cantidad de PEs de memoria."""
    def medir(configuracion):
        if llamadas is not None:
            llamadas.append(configuracion)
        letras = [t for fila in configuracion["malla"] for t in fila.split()]
        area = sum(AREA[t] for t in letras)
        latencia = 1000 / sum(POTENCIA.get(t, 0) for t in letras) + 200 / sum(t in "MG" for t in letras)
        valida = not (invalida and invalida(configuracion["malla"]))
        costo = k_area * area + k_latencia * latencia if valida else 1e6
        return Evaluacion(configuracion["malla"], "correcta" if valida else "incorrecta", costo,
                          {"C_A": area, "C_L_norm": latencia, "C_R": 0.0}, "" if valida else "inválida (falso)")
    return medir


def cotas_falsas(k_area=1.0):
    """Las cotas que corresponden a medir_falso: C ≥ k_area · área, y el área
    mínima de n PEs es un M (12, escribe en memoria) y n - 1 E (1)."""
    def cota(configuracion):
        return k_area * sum(AREA[t] for fila in configuracion["malla"] for t in fila.split())

    def cota_tamano(tipos_pe, pes):
        return k_area * (12 + (pes - 1))
    return cota, cota_tamano


def parametros(**cambios):
    datos = dict(max_rondas=10, lado_maximo=4, iniciales=[[3, 3]], interior_alternativo="C",
                 procesos=1, barrido=False, max_evaluaciones=None, paciencia=None)
    datos.update(cambios)
    return Parametros(**datos)


def registro(nombre, costo, valida=True):
    """Un registro evaluado con ese costo, para probar la cadena a mano."""
    return Registro(((nombre,),), nombre, 0, Evaluacion([], "correcta" if valida else "incorrecta", costo,
                                                        {"C_A": 1}))


class TestAgenteGreedy(unittest.TestCase):
    """El agente explora vecinas en orden fijo y guarda la cadena de mejoras."""

    def test_1_determinismo(self):
        """Dos corridas completas deben dar la misma cadena y el mismo historial."""
        corridas = []
        for _ in range(2):
            agente = Agente(medir_falso(), BASE, parametros())
            cadena = agente.buscar()
            corridas.append(([r.clave for r in cadena], [(r.clave, r.costo) for r in agente.historial]))
        self.assertEqual(corridas[0], corridas[1])

    def test_2_validacion_previa(self):
        """Lo que la validación previa del RL rechaza se descarta sin llamar al evaluador."""
        llamadas = []
        agente = Agente(medir_falso(llamadas=llamadas), BASE, parametros(lado_maximo=4))
        sin_memoria = (("E", "C", "E"),)
        sin_computo = (("M", "M", "M"),)
        dos_columnas = (("M", "M"), ("M", "M"))
        con_g = (("M", "G", "M"),)
        grande = tuple(tuple("MEEEM") for _ in range(5))
        registros = agente.evaluar([sin_memoria, sin_computo, dos_columnas, con_g, grande], 0)
        self.assertEqual(llamadas, [])
        self.assertEqual([r.motivo for r in registros], [
            "la malla no tiene ningún PE de memoria", "la malla no tiene ningún PE de cómputo",
            "columnas = 2 fuera del rango permitido [3, 32]",
            "PE(0,1) es G y accede a memoria: la memoria solo puede ir en los bordes oeste/este; "
            "el interior es de cómputo (E, V o C)",
            "tamaño 5x5 fuera de los límites (filas 1..4, columnas 1..4)",
        ])
        self.assertTrue(all(r.costo == float("inf") for r in registros))
        self.assertIsNone(agente.motivo_invalida((("M", "C", "M"),)))

    def test_3_vecinas(self):
        """Las vecinas deben ser únicas, sin la propia, en orden estable y solo con tipos de cgra.json."""
        agente = Agente(medir_falso(), BASE, parametros())
        vecinas = agente.vecinas(MALLA_3X3)
        self.assertEqual(vecinas, agente.vecinas(MALLA_3X3))
        self.assertEqual(len(vecinas), len(set(vecinas)))
        self.assertNotIn(MALLA_3X3, vecinas)
        self.assertTrue({t for v in vecinas for fila in v for t in fila} <= set(BASE["tipos_pe"]))
        # Primero los movimientos por fila (la fila 0 pasa de E a V; a E sería ella misma).
        self.assertEqual(vecinas[0], (("M", "V", "M"), ("M", "V", "M"), ("M", "C", "M")))
        # Agregar una fila copia la interior más cercana al borde sur, antes de la última.
        self.assertIn((("M", "E", "M"), ("M", "V", "M"), ("M", "V", "M"), ("M", "C", "M")), vecinas)
        # Quitar la columna del medio, y mover el M de (0, 0) al siguiente lugar del perímetro.
        self.assertIn((("M", "M"), ("M", "M"), ("M", "M")), vecinas)
        self.assertIn((("E", "M", "M"), ("M", "V", "M"), ("M", "C", "M")), vecinas)

    def test_4_comparar_con_la_ultima(self):
        """La de C mayor se descarta, la de C menor o igual entra y las siguientes se comparan con ella."""
        agente = Agente(medir_falso(), BASE, parametros())
        a, peor, b, empate, c = registro("a", 5), registro("peor", 6), registro("b", 4), registro("empate", 4), \
            registro("c", 3)
        self.assertEqual(agente.actualizar_cadena([a, peor, b]), [a, b])
        self.assertEqual(agente.actualizar_cadena([registro("entre", 4.5), empate, c, registro("d", 3.5)]),
                         [empate, c])
        self.assertEqual([r.clave for r in agente.cadena], ["a", "b", "empate", "c"])
        self.assertEqual(agente.actualizar_cadena([registro("invalida", 1, valida=False)]), [])

    def test_5_la_cadena_baja_y_termina_en_la_mejor(self):
        """Cada una de la cadena cuesta menos o igual que la anterior, y la última es la de menor C evaluada."""
        agente = Agente(medir_falso(), BASE, parametros())
        cadena = agente.buscar()
        self.assertGreater(len(cadena), 1)
        costos = [r.costo for r in cadena]
        self.assertEqual(costos, sorted(costos, reverse=True))
        self.assertEqual(cadena[-1].costo, min(r.costo for r in agente.historial if r.valida))

    def test_6_sensibilidad_a_k(self):
        """Priorizando área la mejor debe tener menos área; priorizando latencia, menos latencia."""
        por_area = Agente(medir_falso(k_area=100), BASE, parametros()).buscar()[-1]
        por_latencia = Agente(medir_falso(k_latencia=100), BASE, parametros()).buscar()[-1]
        self.assertLess(por_area.area, por_latencia.area)
        self.assertLess(por_latencia.evaluacion.latencia, por_area.evaluacion.latencia)

    def test_7_penalizacion(self):
        """Ninguna configuración inválida debe entrar a la cadena."""
        con_v = lambda malla: any("V" in fila for fila in malla)
        agente = Agente(medir_falso(invalida=con_v), BASE, parametros())
        cadena = agente.buscar()
        self.assertTrue(any(r.evaluacion is not None and not r.valida for r in agente.historial))
        self.assertTrue(cadena)
        self.assertTrue(all(r.valida for r in cadena))

    def test_8_convergencia(self):
        """La cadena solo crece: cada ronda conserva la anterior y su última nunca cuesta más."""
        agente = Agente(medir_falso(), BASE, parametros())
        agente.buscar()
        rondas = agente.cadena_por_ronda
        self.assertGreater(len(rondas), 1)
        for anterior, nueva in zip(rondas, rondas[1:]):
            self.assertEqual(nueva[:len(anterior)], anterior)
            self.assertLessEqual(nueva[-1].costo, anterior[-1].costo)

    def test_9_configuracion_para_el_evaluador(self):
        """La configuración que recibe el evaluador es cgra.json con tamaño fijo y malla explícita."""
        configuracion = Agente(medir_falso(), BASE, parametros()).configuracion(MALLA_3X3)
        self.assertEqual((configuracion["filas"], configuracion["columnas"]), (3, 3))
        self.assertEqual(configuracion["malla"], ["M E M", "M V M", "M C M"])
        self.assertNotIn("distribucion", configuracion)
        self.assertEqual(configuracion["tipos_pe"], BASE["tipos_pe"])

    def test_10_barrido_de_todos_los_tamanos(self):
        """Sin cotas, la ronda 0 revisa todos los tamaños de los límites, de menos a más PEs."""
        agente = Agente(medir_falso(), BASE, parametros(barrido=True, max_rondas=0))
        agente.buscar()
        self.assertEqual(agente.barrido["revisados"], 16)
        self.assertEqual(agente.barrido["sin_revisar"], 0)
        pes = [len(r.malla) * len(r.malla[0]) for r in agente.historial]
        self.assertEqual(pes, sorted(pes))
        self.assertEqual({(len(r.malla), len(r.malla[0])) for r in agente.historial},
                         {(f, c) for f in range(1, 5) for c in range(1, 5)})

    def test_12_empieza_en_la_referencia(self):
        """Con iniciales = 'referencia', la ronda 0 es M E M / M V M / M E M y es la primera de la cadena."""
        agente = Agente(medir_falso(), BASE, parametros(iniciales="referencia", max_rondas=1))
        cadena = agente.buscar()
        self.assertEqual(agente.historial[0].malla, (("M", "E", "M"), ("M", "V", "M"), ("M", "E", "M")))
        self.assertEqual(cadena[0].malla, agente.historial[0].malla)

    def test_13_presupuesto_de_evaluaciones(self):
        """max_evaluaciones cuenta las evaluaciones sin la referencia, como max_iteraciones del RL."""
        llamadas = []
        agente = Agente(medir_falso(llamadas=llamadas), BASE,
                        parametros(iniciales="referencia", max_evaluaciones=5))
        agente.buscar()
        evaluadas = [r for r in agente.historial if r.evaluacion is not None]
        self.assertEqual(len(evaluadas), 6)
        self.assertEqual(len(llamadas), 6)
        self.assertEqual(agente.terminada, "max_evaluaciones")

    def test_14_paciencia(self):
        """Termina tras ``paciencia`` evaluaciones seguidas que no bajan el mínimo (más que la tolerancia)."""
        referencia = ["M E M", "M V M", "M E M"]

        def peor_que_la_referencia(configuracion):
            costo = 1.0 if configuracion["malla"] == referencia else 2.0
            return Evaluacion(configuracion["malla"], "correcta", costo, {"C_A": 1})

        agente = Agente(peor_que_la_referencia, BASE, parametros(iniciales="referencia", paciencia=3))
        agente.buscar()
        self.assertEqual(agente.terminada, "paciencia")
        self.assertEqual(sum(r.evaluacion is not None for r in agente.historial), 1 + 3)

    def test_15_referencia_no_valida(self):
        """Si la referencia no es correcta, la búsqueda se detiene (no hay C_best,0)."""
        agente = Agente(medir_falso(invalida=lambda malla: True), BASE, parametros(iniciales="referencia"))
        self.assertEqual(agente.buscar(), [])
        self.assertEqual(agente.terminada, "referencia no válida")
        self.assertEqual(len(agente.historial), 1)

    def test_11_las_cotas_no_cambian_la_cadena(self):
        """Con las cotas del área, el barrido corta antes y se mide menos, pero la mejor es la misma."""
        resultados = {}
        for con_cotas in (False, True):
            llamadas = []
            cota, cota_tamano = cotas_falsas(k_area=30) if con_cotas else (None, None)
            agente = Agente(medir_falso(k_area=30, llamadas=llamadas), BASE,
                            parametros(barrido=True, lado_maximo=8, max_rondas=3),
                            cota=cota, cota_tamano=cota_tamano)
            cadena = agente.buscar()
            resultados[con_cotas] = ([(r.clave, r.costo) for r in cadena], len(llamadas), agente.barrido)
        self.assertEqual(resultados[True][0], resultados[False][0])
        self.assertLess(resultados[True][1], resultados[False][1])
        self.assertEqual(resultados[False][2]["sin_revisar"], 0)
        self.assertGreater(resultados[True][2]["sin_revisar"], 0)


class TestAgenteConEvaluador(_ConOperaciones):
    """El agente con el evaluador real (Dijkstra + simulador), en una búsqueda corta."""

    def test_1_busqueda_corta_con_el_evaluador_real(self):
        """Desde la referencia, una ronda con el simulador debe dar una cadena correcta que no sube el costo."""
        evaluador = Evaluador([self.pequena], carpeta_programas=self.programas)
        agente = Agente(evaluador.evaluar, BASE,
                        parametros(max_rondas=1, lado_maximo=3, iniciales="referencia", max_evaluaciones=4),
                        cota=evaluador.cota, cota_tamano=evaluador.cota_tamano)
        cadena = agente.buscar()
        self.assertTrue(cadena)
        for r in cadena:
            self.assertTrue(r.valida, r.motivo)
            self.assertEqual(r.evaluacion.correctitud, 0)
        self.assertEqual(cadena[0].malla, agente.referencia())
        self.assertLessEqual(cadena[-1].costo, evaluador.evaluar(evaluador.referencia()).costo)

    def test_2_en_paralelo_da_lo_mismo(self):
        """Evaluar en varios procesos debe dar los mismos costos y en el mismo orden que en serie."""
        agente = Agente(lambda c: None, BASE, parametros())
        configuraciones = [agente.configuracion(m) for m in agente.vecinas(MALLA_3X3)[:6]]
        serie = Evaluador([self.pequena], carpeta_programas=self.programas)
        paralelo = EvaluadorParalelo([self.pequena], self.programas, RUTA_COSTOS, procesos=2)
        try:
            resultados = paralelo.lote(configuraciones)
        finally:
            paralelo.cerrar()
        self.assertEqual([r.costo for r in resultados], [serie.evaluar(c).costo for c in configuraciones])



class TestCompilarSkill02(unittest.TestCase):
    """El agente corre el skill 02 solo cuando falta algo o operaciones.c cambió."""

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
            self.assertEqual(sorted(p.name for p in carpeta.iterdir()), ["memoria.bin", "operaciones.ll"])
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
        self.assertFalse((self.programas / "no_existe" / "operaciones.out").exists())


if __name__ == "__main__":
    unittest.main()
