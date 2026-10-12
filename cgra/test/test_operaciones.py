"""Pruebas de operaciones/operaciones.c: cada operación debe generar su
memoria.bin con las regiones que usan los programas de los PEs."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))

from memoria import ImagenMemoria

FUENTE = os.path.join(_raiz_proyecto, "operaciones", "operaciones.c")
CLANG = shutil.which("clang-18") or shutil.which("clang")


@unittest.skipIf(CLANG is None, "no hay clang-18 para compilar operaciones.c")
class TestOperaciones(unittest.TestCase):
    """operaciones.c tiene las tres operaciones y genera la memoria de la que se pida."""

    @classmethod
    def setUpClass(cls):
        cls.directorio = tempfile.TemporaryDirectory()
        cls.ejecutable = os.path.join(cls.directorio.name, "operaciones.out")
        subprocess.run([CLANG, FUENTE, "-o", cls.ejecutable], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directorio.cleanup()

    def generar(self, operacion):
        """Corre la operación en una carpeta vacía y devuelve su memoria.bin."""
        carpeta = tempfile.mkdtemp(dir=self.directorio.name)
        subprocess.run([self.ejecutable, operacion], cwd=carpeta, check=True, capture_output=True)
        return ImagenMemoria.leer(os.path.join(carpeta, "memoria.bin")).regiones()

    def test_1_reduccion(self):
        """reduccion debe guardar a, b y c (1 x 100, con c en cero) y result (1 x 1) en cero."""
        memoria, formas = self.generar("reduccion")
        self.assertEqual(formas, {"a": (1, 100), "b": (1, 100), "c": (1, 100), "result": (1, 1)})
        self.assertEqual(memoria["a"][:5], [1, 2, 3, 4, 1])
        self.assertEqual(memoria["c"], [0] * 100)
        self.assertEqual(memoria["result"], [0])

    def test_2_matmul(self):
        """matmul debe guardar a, b y result (9 x 9), con result en cero."""
        memoria, formas = self.generar("matmul")
        self.assertEqual(formas, {"a": (9, 9), "b": (9, 9), "result": (9, 9)})
        self.assertEqual(memoria["a"][:3], [1, 2, 3])
        self.assertEqual(memoria["result"], [0] * 81)

    def test_3_convolucion(self):
        """convolucion debe guardar imagen (12 x 12), kernel (4 x 4) y result (9 x 9) en cero."""
        memoria, formas = self.generar("convolucion")
        self.assertEqual(formas, {"imagen": (12, 12), "kernel": (4, 4), "result": (9, 9)})
        self.assertEqual(memoria["result"], [0] * 81)

    def test_4_rechaza_una_operacion_desconocida(self):
        """Con una operación que no existe, el programa debe fallar, listar las operaciones y no escribir memoria.bin."""
        carpeta = tempfile.mkdtemp(dir=self.directorio.name)
        salida = subprocess.run([self.ejecutable, "fft"], cwd=carpeta, capture_output=True, text=True)
        self.assertNotEqual(salida.returncode, 0)
        self.assertIn("reduccion matmul convolucion", salida.stderr)
        self.assertFalse(os.path.exists(os.path.join(carpeta, "memoria.bin")))


if __name__ == "__main__":
    unittest.main()
