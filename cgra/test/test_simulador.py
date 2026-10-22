"""Pruebas del simulador completo: correr una operación y correr todas."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

_raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_raiz_proyecto, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datos_prueba import escribir_reduccion_2x2
from run_cgra import SimuladorCGRA


class TestSimulador(unittest.TestCase):
    """SimuladorCGRA corre una operación o todas las de operaciones.c."""

    def setUp(self):
        self.directorio = tempfile.TemporaryDirectory()
        raiz = self.directorio.name
        self.programas = os.path.join(raiz, "programas")
        # La reducción 2x2 compilada, como la dejan los skills.
        escribir_reduccion_2x2(os.path.join(self.programas, "reduccion"))
        self.formato = os.path.join(raiz, "cgra.json")
        with open(self.formato, "w", encoding="utf-8") as archivo:
            json.dump({"filas": 2, "columnas": 2, "palabras_memoria": 64}, archivo)
        # Una tabla OPERACIONES con una operación que no fue compilada.
        self.fuente = os.path.join(raiz, "operaciones.c")
        with open(self.fuente, "w", encoding="utf-8") as archivo:
            archivo.write('static const Operacion OPERACIONES[] = {\n'
                          '    {"reduccion", reduccion},\n'
                          '    {"sin_compilar", sin_compilar},\n'
                          '};\n')
        self.simulador = SimuladorCGRA(self.formato, self.programas)

    def tearDown(self):
        self.directorio.cleanup()

    def test_1_lee_las_operaciones_de_operaciones_c(self):
        """La lista de operaciones debe salir de la tabla OPERACIONES de operaciones.c, en su orden."""
        self.assertEqual(SimuladorCGRA.operaciones(), ["reduccion", "matmul", "convolucion"])
        self.assertEqual(SimuladorCGRA.operaciones(self.fuente), ["reduccion", "sin_compilar"])

    def test_2_corre_una_operacion_por_su_nombre(self):
        """Con el nombre de una operación debe usar programas/<operacion>/ y devolver cuántos ciclos duró."""
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            ciclos = self.simulador.correr("reduccion")

        self.assertEqual(ciclos, 24)
        self.assertIn("result = 60", salida.getvalue())

    def test_3_corre_todas_y_una_falla_no_detiene_a_las_demas(self):
        """Al correr todas, cada operación debe ejecutarse por separado; si una no está compilada, se anota su error y el resumen muestra las dos."""
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            resultados = self.simulador.correr_todas(fuente=self.fuente)

        self.assertEqual(resultados["reduccion"], 24)
        self.assertIn("No existe", resultados["sin_compilar"])
        texto = salida.getvalue()
        self.assertIn("RESUMEN", texto)
        self.assertRegex(texto, r"reduccion +ok +24 ciclos")
        self.assertRegex(texto, r"sin_compilar +error")

    def test_4_reporte_completo_con_resultados_y_traza_integrada(self):
        """El archivo y la terminal deben presentar las siete secciones y los valores reales, sin duplicar la traza."""
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self.simulador.correr("reduccion", mostrar_traza=True, mostrar_ciclos=True)
        ruta = os.path.join(self.programas, "reduccion", "reporte_ciclos.txt")
        with open(ruta, encoding="utf-8") as archivo:
            reporte = archivo.read()
        for titulo in ("1. CONFIGURACIÓN", "2. RESULTADO", "3. RESUMEN DE LA EJECUCIÓN",
                       "4. DISTRIBUCIÓN DEL TRABAJO", "5. ACTIVIDAD POR PE",
                       "6. EJECUCIÓN CICLO POR CICLO", "7. MEMORIA FINAL"):
            self.assertEqual(reporte.count(titulo), 1)
        self.assertIn("result = 60", reporte)
        self.assertIn("result[0] = 60", reporte)
        self.assertIn("Comparación con referencia: no realizada", reporte)
        self.assertIn("Valores observados (traza):", reporte)
        self.assertIn("[0..9] 2 4 6 8 2 4 6 8 2 4", reporte)
        self.assertIn(reporte.rstrip(), salida.getvalue())
        self.assertNotIn("TRAZA: ciclo | PE", salida.getvalue())

    def test_5_sin_traza_no_inventa_valores_observados(self):
        """Una corrida sin --traza también guarda resultados, pero no efectos por instrucción."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.simulador.correr("reduccion")
        ruta = os.path.join(self.programas, "reduccion", "reporte_ciclos.txt")
        with open(ruta, encoding="utf-8") as archivo:
            reporte = archivo.read()
        self.assertIn("result[0] = 60", reporte)
        self.assertNotIn("Valores observados (traza):", reporte)


if __name__ == "__main__":
    unittest.main()
