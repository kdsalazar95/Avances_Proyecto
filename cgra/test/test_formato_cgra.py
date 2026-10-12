"""Pruebas del formato de la CGRA (cgra.json): tamaño de la malla y de la memoria."""

import json
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from cgra import CGRA
from formato_cgra import FormatoCGRA


class TestFormatoCGRA(unittest.TestCase):
    """cgra.json describe el tamaño de la malla y de la memoria de datos."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directorio.cleanup()

    def escribir_json(self, **cambios):
        datos = {"filas": 2, "columnas": 3, "palabras_memoria": 16}
        datos.update(cambios)
        ruta = os.path.join(self.directorio.name, "cgra.json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)
        return ruta

    def test_1_src_cgra_json_es_valido(self):
        """src/cgra.json, el que leen run_cgra.py y los skills, debe cargarse sin errores y dar una malla y una memoria con al menos un elemento."""
        formato = FormatoCGRA.cargar()
        self.assertGreater(formato.filas, 0)
        self.assertGreater(formato.columnas, 0)
        self.assertGreater(formato.palabras_memoria, 0)

    def test_2_la_cgra_toma_su_tamano_del_formato(self):
        """Al encender la CGRA, la malla debe tener filas x columnas PEs y la memoria el tamaño del formato."""
        cgra = CGRA(FormatoCGRA.cargar(self.escribir_json()))

        self.assertEqual((cgra.filas, cgra.columnas), (2, 3))
        self.assertEqual(len(cgra.malla.pes()), 6)
        self.assertEqual(cgra.malla.ids(), ("PE00", "PE01", "PE02", "PE10", "PE11", "PE12"))
        self.assertEqual(len(cgra.memoria), 16)

    def test_3_rechaza_un_formato_incompleto(self):
        """Si falta filas, columnas o palabras_memoria, el formato no se debe cargar, y el mensaje debe decir qué falta."""
        ruta = self.escribir_json()
        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        del datos["palabras_memoria"]
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)

        with self.assertRaisesRegex(ValueError, "faltan los campos palabras_memoria"):
            FormatoCGRA.cargar(ruta)

    def test_4_rechaza_tamanos_invalidos(self):
        """Una malla sin PEs, una memoria sin palabras o un tamaño que no es entero deben fallar al cargar el formato."""
        with self.assertRaisesRegex(ValueError, "al menos una fila"):
            FormatoCGRA.cargar(self.escribir_json(filas=0))
        with self.assertRaisesRegex(ValueError, "al menos una palabra"):
            FormatoCGRA.cargar(self.escribir_json(palabras_memoria=0))
        with self.assertRaisesRegex(ValueError, "columnas deben ser números enteros"):
            FormatoCGRA.cargar(self.escribir_json(columnas="4"))

    def test_5_rechaza_un_json_que_no_existe_o_esta_mal_escrito(self):
        """Un formato que no existe o que no es JSON válido debe fallar con un mensaje claro."""
        with self.assertRaisesRegex(ValueError, "No existe el formato"):
            FormatoCGRA.cargar(os.path.join(self.directorio.name, "no_existe.json"))

        ruta = os.path.join(self.directorio.name, "roto.json")
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write("{filas: 4")
        with self.assertRaisesRegex(ValueError, "JSON inválido"):
            FormatoCGRA.cargar(ruta)


if __name__ == "__main__":
    unittest.main()
