"""Pruebas del arranque de la CGRA (memoria en blanco), del cargador, que
copia la imagen de memoria y prepara el programa, y de la traza."""

import contextlib
import io
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cargador import Cargador
from cgra import CGRA
from datos_prueba import escribir_reduccion_2x2
from formato_cgra import FormatoCGRA
from programa import Programa
from traza import Traza


class TestCargador(unittest.TestCase):
    """Encender la CGRA y cargarle un programa, como hace el sistema operativo."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()
        # Imagen de la reducción 2x2: 12 + 12 + 12 + 1 = 37 palabras.
        self.ruta_memoria, self.ruta_instrucciones = escribir_reduccion_2x2(self.directorio.name)

    def tearDown(self):
        self.directorio.cleanup()

    def test_1_la_cgra_arranca_en_blanco(self):
        """Al encenderse, la CGRA debe tener toda la memoria en cero, sin regiones cargadas, y sus PEs con los registros en cero."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))

        self.assertEqual(cgra.memoria.palabras, [0] * 64)
        self.assertEqual(cgra.memoria.simbolos, {})
        for pe_id in cgra.malla.ids():
            pe = cgra.malla.pe(pe_id)
            self.assertEqual(pe.registros, [0] * 16)

    def test_2_copia_la_imagen_y_el_resto_queda_en_cero(self):
        """El cargador debe copiar cada región a su dirección dentro de una memoria más grande, y el resto de la memoria debe seguir en cero."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))
        Cargador(cgra).cargar_imagen(self.ruta_memoria)

        self.assertEqual(cgra.memoria.simbolos["a"], (0, 1, 12))
        self.assertEqual(cgra.memoria.simbolos["result"], (36, 1, 1))
        self.assertEqual(cgra.memoria.palabras[0:4], [1, 2, 3, 4])
        self.assertEqual(cgra.memoria.palabras[37:], [0] * (64 - 37))

    def test_3_la_imagen_se_puede_cargar_en_otra_direccion(self):
        """Con una dirección base distinta de 0, las regiones deben quedar desplazadas y la memoria anterior a la base en cero."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))
        Cargador(cgra).cargar_imagen(self.ruta_memoria, base=10)

        self.assertEqual(cgra.memoria.simbolos["a"], (10, 1, 12))
        self.assertEqual(cgra.memoria.palabras[:10], [0] * 10)
        self.assertEqual(cgra.memoria.region("a"), [1, 2, 3, 4] * 3)

    def test_4_rechaza_una_imagen_que_no_cabe(self):
        """Si la imagen es más grande que la memoria de la CGRA, el cargador debe fallar en vez de cortarla."""
        cgra = CGRA(FormatoCGRA(2, 2, 30))
        with self.assertRaisesRegex(ValueError, "ocupa 37 palabras"):
            Cargador(cgra).cargar_imagen(self.ruta_memoria)

    def test_5_enlaza_ld_st_a_direcciones(self):
        """El enlace debe convertir cada banco[índice] de LD/ST en la dirección de esa palabra, y rechazar índices fuera de la región o regiones que no existen."""
        simbolos = {"a": (0, 1, 12), "result": (36, 1, 1)}
        programa = Programa({"PE00": {0: {"op": "ld", "dst": "rA", "bank": "a", "index": 3},
                                      1: {"op": "st", "bank": "result", "index": 0, "src": "acc"}}})
        programa.enlazar(simbolos)
        self.assertEqual(programa["PE00"][0]["addr"], 3)
        self.assertEqual(programa["PE00"][1]["addr"], 36)

        with self.assertRaisesRegex(ValueError, r"a\[12\] se sale"):
            Programa({"PE00": {0: {"op": "ld", "dst": "rA", "bank": "a", "index": 12}}}).enlazar(simbolos)
        with self.assertRaisesRegex(ValueError, "'z' no está"):
            Programa({"PE00": {0: {"op": "ld", "dst": "rA", "bank": "z", "index": 0}}}).enlazar(simbolos)

    def test_6_carga_el_programa_y_se_ejecuta(self):
        """Después de cargar debe haber un programa de 24 instrucciones por PE, y al ejecutar la reducción 2x2 debe quedar result = 60 en la memoria de la CGRA."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))
        programa = Cargador(cgra).cargar(self.ruta_memoria, self.ruta_instrucciones)
        self.assertEqual(set(programa.instrucciones), set(cgra.malla.ids()))
        self.assertEqual(len(programa.ciclos()), 24)

        cgra.ejecutar(programa)

        self.assertEqual(cgra.memoria.region("c"), [2, 4, 6, 8] * 3)
        self.assertEqual(cgra.memoria.region("result"), [60])
        self.assertEqual(cgra.memoria.palabras[37:], [0] * (64 - 37))

    def test_7_la_traza_muestra_instrucciones_sin_saber_la_operacion(self):
        """Con la traza, cada línea debe mostrar el ciclo, el PE, la instrucción y lo que escribió (sin los NOP), sin mencionar qué operación se está corriendo."""
        cgra = CGRA(FormatoCGRA(2, 2, 64))
        traza = Traza()
        cgra.ejecutar(Cargador(cgra).cargar(self.ruta_memoria, self.ruta_instrucciones), traza)
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            traza.mostrar()
        texto = salida.getvalue()

        self.assertRegex(texto, r"1 \| PE00 \| LD rA, a\[0\] +\| rA = 1")
        self.assertRegex(texto, r"19 \| PE01 \| SEND west, acc +\| envía 12")
        self.assertRegex(texto, r"23 \| PE00 \| ST result\[0\], acc +\| result\[0\] = 60  \(dirección 36\)")
        self.assertNotIn("NOP", texto)
        self.assertNotIn("reduc", texto.lower())


if __name__ == "__main__":
    unittest.main()
