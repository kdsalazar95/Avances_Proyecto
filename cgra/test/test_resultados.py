"""Pruebas de la memoria binaria de la CGRA (memoria.bin) y de la
presentación del resultado, que no depende de la operación."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from memoria import ImagenMemoria
from resultados import Resultados


class TestMemoriaBinaria(unittest.TestCase):
    """Pruebas de ImagenMemoria: guardar y leer memoria.bin."""

    def test_1_guarda_y_carga_las_regiones_con_su_forma(self):
        """Una memoria guardada en binario debe volver a cargarse con los
        mismos bancos, valores y forma (filas x columnas) de cada región."""
        memoria = {"a": [1, 2, 3, 4], "b": [0.5, 1.5, 2.5, 3.5], "result": [0]}
        formas = {"a": (2, 2), "b": (2, 2), "result": (1, 1)}
        with tempfile.TemporaryDirectory() as directorio:
            ruta = os.path.join(directorio, "memoria.bin")
            ImagenMemoria.desde_regiones(memoria, formas).guardar(ruta)
            cargada, formas_cargadas = ImagenMemoria.leer(ruta).regiones()

        self.assertEqual(cargada, memoria)
        self.assertEqual(formas_cargadas, formas)
        self.assertEqual(list(cargada), ["a", "b", "result"])

    def test_2_rechaza_un_archivo_que_no_es_memoria_de_la_cgra(self):
        """Un archivo sin la firma CGRAMEM debe fallar en vez de cargarse
        como datos basura."""
        with tempfile.TemporaryDirectory() as directorio:
            ruta = os.path.join(directorio, "memoria.bin")
            with open(ruta, "wb") as archivo:
                archivo.write(b"NOESMEMORIA" + bytes(16))
            with self.assertRaisesRegex(ValueError, "no es una memoria de la CGRA"):
                ImagenMemoria.leer(ruta)

    def test_3_rechaza_una_region_que_no_coincide_con_su_forma(self):
        """Si un banco no tiene filas x columnas valores, no se debe
        guardar una memoria inconsistente."""
        with tempfile.TemporaryDirectory() as directorio:
            ruta = os.path.join(directorio, "memoria.bin")
            with self.assertRaisesRegex(ValueError, "forma es 2x2"):
                ImagenMemoria.desde_regiones({"a": [1, 2, 3]}, {"a": (2, 2)}).guardar(ruta)


class TestPresentacionResultado(unittest.TestCase):
    """Pruebas de Resultados, que solo mira la memoria."""

    def mostrar(self, inicial, final, formas):
        salida = io.StringIO()
        with redirect_stdout(salida):
            Resultados(inicial, final, formas).mostrar()
        return salida.getvalue()

    def test_1_solo_muestra_las_regiones_que_escribieron_los_pes(self):
        """Las entradas que no cambiaron (a, b) no se muestran; la región
        escrita (result) sí."""
        inicial = {"a": [1, 2], "b": [3, 4], "result": [0]}
        final = {"a": [1, 2], "b": [3, 4], "result": [11]}

        self.assertEqual(Resultados(inicial, final, {}).modificadas(), ["result"])
        texto = self.mostrar(inicial, final, {"a": (1, 2), "b": (1, 2), "result": (1, 1)})
        self.assertIn("result = 11", texto)
        self.assertNotIn("a:", texto)

    def test_2_muestra_una_region_matricial_por_filas(self):
        """Una región 2x2 se debe imprimir organizada en filas, no como una
        lista plana, sin necesidad de saber si fue matmul o convolución."""
        texto = self.mostrar({"result": [0, 0, 0, 0]}, {"result": [19, 22, 43, 50]}, {"result": (2, 2)})

        self.assertIn("[19, 22]", texto)
        self.assertIn("[43, 50]", texto)

    def test_3_avisa_si_los_pes_no_modificaron_la_memoria(self):
        """Si ningún ST cambió la memoria, se debe decir explícitamente."""
        texto = self.mostrar({"a": [1]}, {"a": [1]}, {"a": (1, 1)})

        self.assertIn("no modificaron la memoria", texto)


if __name__ == "__main__":
    unittest.main()
